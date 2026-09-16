#include "check.hpp"
#include "coarsen.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>

namespace gpart {
namespace {
class GpuEventTimer {
public:
    GpuEventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
        CUDA_CHECK(cudaEventRecord(start_));
    }

    ~GpuEventTimer() {
        cudaEventDestroy(stop_);
        cudaEventDestroy(start_);
    }

    double seconds() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return static_cast<double>(milliseconds) * 1.0e-3;
    }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

__device__ __forceinline__ std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

template <typename OffsetT, typename WeightT>
__global__ void weighted_cut_kernel(
    std::int64_t n, const OffsetT* offsets,
    const std::int32_t* neighbors, const WeightT* edge_weights,
    const std::int32_t* labels, unsigned long long* cut) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    unsigned long long local = 0;
    const int p = labels[v];
    for (auto e = offsets[v]; e < offsets[v + 1]; ++e) {
        if (labels[neighbors[e]] != p) local += edge_weights[e];
    }
    if (local) atomicAdd(cut, local);
}

constexpr int WARPS_PER_BLOCK = 8;
constexpr int WARP_SIZE = 32;
constexpr std::uint64_t INVALID_KEY = std::numeric_limits<std::uint64_t>::max();

template <typename WeightT>
__device__ __forceinline__ void atomic_add_graph_weight(
    WeightT* address, WeightT value) {
    if constexpr (sizeof(WeightT) == sizeof(std::uint32_t)) {
        atomicAdd(reinterpret_cast<unsigned int*>(address),
                  static_cast<unsigned int>(value));
    } else {
        atomicAdd(reinterpret_cast<unsigned long long*>(address),
                  static_cast<unsigned long long>(value));
    }
}

template <typename WeightT>
__global__ void coarse_vertex_weights_kernel(
    std::int64_t n, const WeightT* vertex_weights,
    const std::int32_t* map, WeightT* coarse_weights) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    atomic_add_graph_weight(coarse_weights + map[v], vertex_weights[v]);
}

template <typename OffsetT, typename WeightT>
__global__ void compact_cross_edges_warp_kernel(
    std::int64_t n, const OffsetT* offsets,
    const std::int32_t* neighbors, const WeightT* edge_weights,
    const std::int32_t* map, std::uint64_t* keys, std::uint64_t* values,
    unsigned long long* cross_count) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      WARPS_PER_BLOCK + (threadIdx.x / WARP_SIZE);
    if (warp >= n) return;
    const auto source = static_cast<std::uint32_t>(map[warp]);
    const auto begin = offsets[warp];
    const auto end = offsets[warp + 1];
    for (auto base_edge = begin; base_edge < end; base_edge += WARP_SIZE) {
        const auto e = base_edge + lane;
        const auto target = e < end
            ? static_cast<std::uint32_t>(map[neighbors[e]]) : source;
        const bool cross = e < end && source != target;
        const unsigned mask = __ballot_sync(0xffffffffU, cross);
        unsigned long long output_base = 0;
        if (lane == 0 && mask) {
            output_base = atomicAdd(
                cross_count, static_cast<unsigned long long>(__popc(mask)));
        }
        output_base = __shfl_sync(0xffffffffU, output_base, 0);
        if (cross) {
            const auto before = mask & ((1U << lane) - 1U);
            const auto output = output_base + __popc(before);
            keys[output] = (static_cast<std::uint64_t>(source) << 32) | target;
            values[output] = edge_weights[e];
        }
    }
}

template <typename OffsetT, typename WeightT>
__global__ void write_coarse_csr_kernel(
    std::int64_t count, const std::uint64_t* keys,
    const std::uint64_t* values, std::int32_t* neighbors,
    WeightT* edge_weights, OffsetT* offsets) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto source = static_cast<std::uint32_t>(keys[i] >> 32);
    neighbors[i] = static_cast<std::int32_t>(keys[i]);
    edge_weights[i] = static_cast<WeightT>(values[i]);
    if (i == 0 || source != static_cast<std::uint32_t>(keys[i - 1] >> 32)) {
        offsets[source] = static_cast<OffsetT>(i);
    }
}
constexpr int SCLP_LOW_DEGREE = 8;
constexpr int SCLP_MEDIUM_DEGREE = 256;
constexpr int SCLP_AFFINITY_WARPS_PER_BLOCK = 4;
constexpr int SCLP_MEDIUM_HASH_SIZE = 512;
constexpr std::int32_t SCLP_INVALID = 0x7fffffff;

template <typename OffsetT>
__global__ void sclp_degree_class_kernel(
    std::int64_t n, const OffsetT* offsets,
    std::int32_t* low_flags, std::int32_t* medium_flags,
    std::int32_t* high_flags) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const auto degree = offsets[v + 1] - offsets[v];
    low_flags[v] = degree <= SCLP_LOW_DEGREE ? 1 : 0;
    medium_flags[v] = degree > SCLP_LOW_DEGREE &&
        degree <= SCLP_MEDIUM_DEGREE ? 1 : 0;
    high_flags[v] = degree > SCLP_MEDIUM_DEGREE ? 1 : 0;
}

__device__ __forceinline__ bool sclp_cluster_is_mover(
    std::int32_t cluster, std::uint32_t role_salt,
    std::uint32_t mover_threshold) {
    return mix32(static_cast<std::uint32_t>(cluster) ^ role_salt) < mover_threshold;
}

__device__ __forceinline__ bool sclp_better_candidate(
    unsigned long long affinity, unsigned long long tie,
    unsigned long long other_affinity, unsigned long long other_tie) {
    return affinity > other_affinity ||
           (affinity == other_affinity && affinity != 0 && tie > other_tie);
}

__device__ __forceinline__ void sclp_update_best(
    unsigned long long affinity, unsigned long long tie,
    unsigned long long& best_affinity, unsigned long long& best_tie) {
    if (sclp_better_candidate(affinity, tie, best_affinity, best_tie)) {
        best_affinity = affinity;
        best_tie = tie;
    }
}

__device__ __forceinline__ void sclp_finish_warp_affinity(
    std::int32_t v, unsigned long long local_current,
    unsigned long long local_unrestricted,
    unsigned long long local_best_affinity, unsigned long long local_best_tie,
    unsigned long long* current_affinity,
    unsigned long long* best_affinity, unsigned long long* best_ties,
    unsigned long long* unrestricted_best_affinity) {
    constexpr unsigned mask = 0xffffffffU;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_current = max(
            local_current, __shfl_down_sync(mask, local_current, offset));
        local_unrestricted = max(
            local_unrestricted,
            __shfl_down_sync(mask, local_unrestricted, offset));
        const auto other_affinity =
            __shfl_down_sync(mask, local_best_affinity, offset);
        const auto other_tie = __shfl_down_sync(mask, local_best_tie, offset);
        if (sclp_better_candidate(
                other_affinity, other_tie,
                local_best_affinity, local_best_tie)) {
            local_best_affinity = other_affinity;
            local_best_tie = other_tie;
        }
    }
    if (lane == 0) {
        current_affinity[v] = local_current;
        unrestricted_best_affinity[v] = local_unrestricted;
        best_affinity[v] = local_best_affinity;
        best_ties[v] = local_best_tie;
    }
}

__device__ __forceinline__ void sclp_finish_subwarp_affinity(
    std::int32_t v, unsigned mask, int lane_in_subwarp,
    unsigned long long local_current,
    unsigned long long local_unrestricted,
    unsigned long long local_best_affinity, unsigned long long local_best_tie,
    unsigned long long* current_affinity,
    unsigned long long* best_affinity, unsigned long long* best_ties,
    unsigned long long* unrestricted_best_affinity) {
    constexpr int SUBWARP_SIZE = 8;
    for (int offset = SUBWARP_SIZE / 2; offset > 0; offset /= 2) {
        local_current = max(
            local_current, __shfl_down_sync(mask, local_current, offset));
        local_unrestricted = max(
            local_unrestricted,
            __shfl_down_sync(mask, local_unrestricted, offset));
        const auto other_affinity =
            __shfl_down_sync(mask, local_best_affinity, offset);
        const auto other_tie = __shfl_down_sync(mask, local_best_tie, offset);
        if (sclp_better_candidate(
                other_affinity, other_tie,
                local_best_affinity, local_best_tie)) {
            local_best_affinity = other_affinity;
            local_best_tie = other_tie;
        }
    }
    if (lane_in_subwarp == 0) {
        current_affinity[v] = local_current;
        unrestricted_best_affinity[v] = local_unrestricted;
        best_affinity[v] = local_best_affinity;
        best_ties[v] = local_best_tie;
    }
}

template <typename OffsetT, typename WeightT>
__global__ void sclp_affinity_low_match_kernel(
    std::int64_t count, const std::int32_t* vertices,
    const OffsetT* offsets, const std::int32_t* neighbors,
    const WeightT* edge_weights, const std::int32_t* clusters,
    std::uint32_t role_salt, std::uint32_t mover_threshold,
    std::uint32_t tie_salt, bool filter_roles,
    unsigned long long* current_affinity,
    unsigned long long* best_affinity, unsigned long long* best_ties,
    unsigned long long* unrestricted_best_affinity) {
    constexpr int SUBWARP_SIZE = 8;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int lane_in_subwarp = lane & (SUBWARP_SIZE - 1);
    const int subwarp_in_warp = lane / SUBWARP_SIZE;
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    const auto vertex_index = warp * (WARP_SIZE / SUBWARP_SIZE) +
                              subwarp_in_warp;
    if (vertex_index >= count) return;
    const auto v = vertices[vertex_index];
    const auto begin = offsets[v];
    const auto degree = offsets[v + 1] - begin;
    const int subwarp_begin = subwarp_in_warp * SUBWARP_SIZE;
    const unsigned subwarp_mask =
        ((1U << SUBWARP_SIZE) - 1U) << subwarp_begin;
    const bool valid = lane_in_subwarp < degree &&
                       neighbors[begin + lane_in_subwarp] != v;
    const unsigned active = __ballot_sync(subwarp_mask, valid);
    std::int32_t target = SCLP_INVALID;
    unsigned long long connection = 0;
    bool leader = false;
    if (valid) {
        target = clusters[neighbors[begin + lane_in_subwarp]];
        const unsigned peers = __match_any_sync(active, target);
        leader = lane == (__ffs(peers) - 1);
        const auto my_weight =
            static_cast<unsigned long long>(
                edge_weights[begin + lane_in_subwarp]);
        for (int source_lane = subwarp_begin;
             source_lane < subwarp_begin + SUBWARP_SIZE; ++source_lane) {
            const auto peer_weight =
                __shfl_sync(active, my_weight, source_lane);
            if (leader && (peers & (1U << source_lane))) {
                connection += peer_weight;
            }
        }
    }
    unsigned long long local_current = 0;
    unsigned long long local_unrestricted = 0;
    unsigned long long best_value = 0;
    unsigned long long best_tie = 0;
    const auto source = clusters[v];
    if (leader && target == source) local_current = connection;
    if (leader && target != source) {
        local_unrestricted = connection;
        if (!filter_roles || !sclp_cluster_is_mover(
                target, role_salt, mover_threshold)) {
            const auto hash = mix32(
                static_cast<std::uint32_t>(v) ^
                mix32(static_cast<std::uint32_t>(target)) ^ tie_salt);
            best_value = connection;
            best_tie = (static_cast<unsigned long long>(hash) << 32) |
                       (0xffffffffULL - static_cast<std::uint32_t>(target));
        }
    }
    sclp_finish_subwarp_affinity(
        v, subwarp_mask, lane_in_subwarp,
        local_current, local_unrestricted,
        best_value, best_tie,
        current_affinity, best_affinity, best_ties,
        unrestricted_best_affinity);
}

template <typename OffsetT, typename WeightT>
__global__ void sclp_affinity_medium_hash_kernel(
    std::int64_t count, const std::int32_t* vertices,
    const OffsetT* offsets, const std::int32_t* neighbors,
    const WeightT* edge_weights, const std::int32_t* clusters,
    std::uint32_t role_salt, std::uint32_t mover_threshold,
    std::uint32_t tie_salt, bool filter_roles,
    unsigned long long* current_affinity,
    unsigned long long* best_affinity, unsigned long long* best_ties,
    unsigned long long* unrestricted_best_affinity) {
    extern __shared__ unsigned char storage[];
    auto* all_keys = reinterpret_cast<std::int32_t*>(storage);
    auto* all_values = reinterpret_cast<unsigned long long*>(
        all_keys + SCLP_AFFINITY_WARPS_PER_BLOCK * SCLP_MEDIUM_HASH_SIZE);
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      SCLP_AFFINITY_WARPS_PER_BLOCK + warp_in_block;
    if (warp >= count) return;
    auto* keys = all_keys + warp_in_block * SCLP_MEDIUM_HASH_SIZE;
    auto* values = all_values + warp_in_block * SCLP_MEDIUM_HASH_SIZE;
    for (int slot = lane; slot < SCLP_MEDIUM_HASH_SIZE; slot += WARP_SIZE) {
        keys[slot] = -1;
        values[slot] = 0;
    }
    __syncwarp();
    const auto v = vertices[warp];
    for (auto e = offsets[v] + lane; e < offsets[v + 1]; e += WARP_SIZE) {
        if (neighbors[e] == v) continue;
        const auto target = clusters[neighbors[e]];
        int slot = static_cast<int>(mix32(static_cast<std::uint32_t>(target))) &
                   (SCLP_MEDIUM_HASH_SIZE - 1);
        while (true) {
            const auto found = atomicCAS(keys + slot, -1, target);
            if (found == -1 || found == target) {
                atomicAdd(values + slot,
                          static_cast<unsigned long long>(edge_weights[e]));
                break;
            }
            slot = (slot + 1) & (SCLP_MEDIUM_HASH_SIZE - 1);
        }
    }
    __syncwarp();
    unsigned long long local_current = 0;
    unsigned long long local_unrestricted = 0;
    unsigned long long best_value = 0;
    unsigned long long best_tie = 0;
    const auto source = clusters[v];
    for (int slot = lane; slot < SCLP_MEDIUM_HASH_SIZE; slot += WARP_SIZE) {
        const auto target = keys[slot];
        if (target < 0) continue;
        const auto connection = values[slot];
        if (target == source) {
            local_current = connection;
            continue;
        }
        local_unrestricted = max(local_unrestricted, connection);
        if (filter_roles && sclp_cluster_is_mover(
                target, role_salt, mover_threshold)) continue;
        const auto hash = mix32(
            static_cast<std::uint32_t>(v) ^
            mix32(static_cast<std::uint32_t>(target)) ^ tie_salt);
        const auto tie = (static_cast<unsigned long long>(hash) << 32) |
                         (0xffffffffULL - static_cast<std::uint32_t>(target));
        sclp_update_best(connection, tie, best_value, best_tie);
    }
    sclp_finish_warp_affinity(
        v, local_current, local_unrestricted,
        best_value, best_tie,
        current_affinity, best_affinity, best_ties,
        unrestricted_best_affinity);
}

template <typename OffsetT>
__global__ void sclp_high_edge_counts_kernel(
    std::int64_t count, const std::int32_t* vertices,
    const OffsetT* offsets, std::int64_t* counts) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const auto v = vertices[i];
        counts[i] = offsets[v + 1] - offsets[v];
    }
}

template <typename OffsetT, typename WeightT>
__global__ void sclp_fill_affinity_high_kernel(
    std::int64_t count, const std::int32_t* vertices,
    const std::int64_t* compact_offsets,
    const OffsetT* offsets, const std::int32_t* neighbors,
    const WeightT* edge_weights, const std::int32_t* clusters,
    std::uint64_t* keys, std::uint64_t* values) {
    const auto i = static_cast<std::int64_t>(blockIdx.x);
    if (i >= count) return;
    const auto v = vertices[i];
    const auto begin = offsets[v];
    const auto compact_begin = compact_offsets[i];
    for (auto e = begin + threadIdx.x; e < offsets[v + 1]; e += blockDim.x) {
        const auto output = compact_begin + (e - begin);
        if (neighbors[e] == v) {
            keys[output] = INVALID_KEY;
            values[output] = 0;
            continue;
        }
        keys[output] =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)) << 32) |
            static_cast<std::uint32_t>(clusters[neighbors[e]]);
        values[output] = edge_weights[e];
    }
}

__global__ void sclp_affinity_baseline_kernel(
    std::int64_t count, const std::uint64_t* keys,
    const std::uint64_t* connections, const std::int32_t* clusters,
    std::uint32_t role_salt, std::uint32_t mover_threshold,
    bool filter_roles, unsigned long long* current_affinity,
    unsigned long long* best_affinity,
    unsigned long long* unrestricted_best_affinity) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto v = static_cast<std::uint32_t>(keys[i] >> 32);
    const auto target = static_cast<std::int32_t>(keys[i]);
    const auto source = clusters[v];
    if (target == source) {
        current_affinity[v] = static_cast<unsigned long long>(connections[i]);
        return;
    }
    atomicMax(unrestricted_best_affinity + v,
              static_cast<unsigned long long>(connections[i]));
    if (filter_roles && sclp_cluster_is_mover(
            target, role_salt, mover_threshold)) return;
    atomicMax(best_affinity + v,
              static_cast<unsigned long long>(connections[i]));
}

__global__ void sclp_best_tie_kernel(
    std::int64_t count, const std::uint64_t* keys,
    const std::uint64_t* connections, const std::int32_t* clusters,
    const unsigned long long* best_affinity, std::uint32_t tie_salt,
    std::uint32_t role_salt, std::uint32_t mover_threshold,
    bool filter_roles, unsigned long long* best_ties) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto v = static_cast<std::uint32_t>(keys[i] >> 32);
    const auto target = static_cast<std::int32_t>(keys[i]);
    const auto source = clusters[v];
    if (target == source) return;
    const auto hash = mix32(v ^ mix32(static_cast<std::uint32_t>(target)) ^ tie_salt);
    const auto key = (static_cast<unsigned long long>(hash) << 32) |
                     (0xffffffffULL - static_cast<std::uint32_t>(target));
    if (connections[i] != best_affinity[v] ||
        (filter_roles && sclp_cluster_is_mover(
             target, role_salt, mover_threshold))) return;
    atomicMax(best_ties + v, key);
}

__global__ void sclp_decode_gain_kernel(
    std::int64_t n, const std::int32_t* clusters,
    const unsigned long long* current_affinity,
    const unsigned long long* best_affinity,
    const unsigned long long* best_ties,
    const unsigned long long* unrestricted_best_affinity,
    std::uint32_t role_salt, std::uint32_t mover_threshold,
    bool filter_roles, std::int32_t* proposals,
    unsigned long long* gains, std::uint32_t* proposal_ties,
    unsigned long long* role_diagnostics) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    proposals[v] = SCLP_INVALID;
    gains[v] = 0;
    proposal_ties[v] = 0;
    const auto source = clusters[v];
    const auto unrestricted_gain = unrestricted_best_affinity[v] > current_affinity[v]
        ? unrestricted_best_affinity[v] - current_affinity[v] : 0ULL;
    if (role_diagnostics != nullptr && unrestricted_gain > 0) {
        atomicAdd(role_diagnostics, 1ULL);
    }
    const bool source_can_move = !filter_roles || sclp_cluster_is_mover(
        source, role_salt, mover_threshold);
    const auto role_gain = source_can_move && best_affinity[v] > current_affinity[v]
        ? best_affinity[v] - current_affinity[v] : 0ULL;
    if (role_diagnostics != nullptr && unrestricted_gain > role_gain) {
        atomicAdd(role_diagnostics + 1, 1ULL);
        atomicAdd(role_diagnostics + 2, unrestricted_gain - role_gain);
    }
    if (role_gain == 0) return;
    const auto target = static_cast<std::int32_t>(
        0xffffffffULL - (best_ties[v] & 0xffffffffULL));
    if (target < 0 || target == source) return;
    proposals[v] = target;
    gains[v] = best_affinity[v] - current_affinity[v];
    proposal_ties[v] = static_cast<std::uint32_t>(best_ties[v] >> 32);
}

struct SclpAdmissionKey {
    std::uint32_t target;
    std::uint64_t reverse_gain;
    std::uint32_t reverse_tie;
    std::uint32_t vertex;
};

struct SclpAdmissionKeyDecomposer {
    __host__ __device__ auto operator()(SclpAdmissionKey& key) const {
        return cuda::std::tie(
            key.target, key.reverse_gain, key.reverse_tie, key.vertex);
    }
};

struct SclpValidAdmissionKey {
    __host__ __device__ bool operator()(const SclpAdmissionKey& key) const {
        return key.target != static_cast<std::uint32_t>(SCLP_INVALID);
    }
};

__global__ void sclp_admission_keys_kernel(
    std::int64_t n, const std::int32_t* targets,
    const unsigned long long* gains, const std::uint32_t* ties,
    SclpAdmissionKey* keys) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    keys[v] = {
        static_cast<std::uint32_t>(targets[v]),
        ~static_cast<std::uint64_t>(gains[v]),
        ~ties[v],
        static_cast<std::uint32_t>(v)};
}

template <typename WeightT>
__global__ void sclp_decode_admission_keys_kernel(
    std::int64_t count, const SclpAdmissionKey* keys,
    const WeightT* vertex_weights, std::int32_t* order,
    std::int32_t* targets, std::uint64_t* weights) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto v = static_cast<std::int32_t>(keys[i].vertex);
    order[i] = v;
    targets[i] = static_cast<std::int32_t>(keys[i].target);
    weights[i] = vertex_weights[v];
}

struct SclpNonzeroWeight {
    __host__ __device__ bool operator()(unsigned long long value) const {
        return value != 0;
    }
};

__global__ void sclp_commit_kernel(
    std::int64_t count, const std::int32_t* order,
    const std::int32_t* targets, const std::uint64_t* weights,
    const std::uint64_t* prefix_weights,
    const unsigned long long* base_cluster_weights, std::uint64_t capacity,
    const unsigned long long* gains,
    std::int32_t* clusters, unsigned long long* cluster_weights,
    unsigned long long* accepted, unsigned long long* rejected,
    unsigned long long* predicted_gain) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto target = targets[i];
    const auto base = static_cast<std::uint64_t>(base_cluster_weights[target]);
    const auto prefix = prefix_weights[i];
    const auto v = order[i];
    const auto source = clusters[v];
    const auto weight = static_cast<unsigned long long>(weights[i]);
    if (base <= capacity && prefix <= capacity - base) {
        atomicAdd(cluster_weights + source, 0ULL - weight);
        atomicAdd(cluster_weights + target, weight);
        clusters[v] = target;
        if (accepted != nullptr) {
            atomicAdd(accepted, 1ULL);
            atomicAdd(predicted_gain, gains[v]);
        }
    } else {
        if (rejected != nullptr) atomicAdd(rejected, 1ULL);
    }
}

template <typename WeightT>
__global__ void sclp_singleton_favorite_kernel(
    std::int64_t n, const WeightT* vertex_weights,
    const unsigned long long* cluster_weights, const std::int32_t* clusters,
    const std::int32_t* proposals, std::int32_t* favorites) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const auto source = clusters[v];
    favorites[v] = cluster_weights[source] == vertex_weights[v]
        ? proposals[v] : SCLP_INVALID;
}

__global__ void sclp_pair_flags_kernel(
    std::int64_t count, const std::int32_t* order,
    const std::uint64_t* ordered_weights,
    const std::uint64_t* ranks, std::uint64_t capacity,
    std::int32_t* pair_flags) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const bool second = (ranks[i] & 1ULL) == 0ULL;
    pair_flags[i] = second && i > 0 &&
        ordered_weights[i - 1] <= capacity - ordered_weights[i] ? 1 : 0;
}

__global__ void sclp_two_hop_pair_commit_kernel(
    std::int64_t count, const std::int32_t* order,
    const std::int32_t* pair_flags, const std::int32_t* pair_positions,
    std::uint64_t merge_budget, const std::uint64_t* weights,
    std::int32_t* clusters,
    unsigned long long* cluster_weights, unsigned long long* merged) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count || !pair_flags[i] ||
        static_cast<std::uint64_t>(pair_positions[i]) >= merge_budget) return;
    const auto v = order[i];
    const auto source = clusters[v];
    const auto target = clusters[order[i - 1]];
    if (source == target) return;
    const auto weight = static_cast<unsigned long long>(weights[i]);
    atomicAdd(cluster_weights + source, 0ULL - weight);
    atomicAdd(cluster_weights + target, weight);
    clusters[v] = target;
    if (merged != nullptr) atomicAdd(merged, 1ULL);
}

__global__ void sclp_nonempty_cluster_kernel(
    std::int64_t n, const unsigned long long* cluster_weights,
    std::int32_t* flags) {
    const auto c = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c < n) flags[c] = cluster_weights[c] > 0 ? 1 : 0;
}

__global__ void sclp_compact_map_kernel(
    std::int64_t n, const std::int32_t* clusters,
    const std::int32_t* compact_ids, std::int32_t* map) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v < n) map[v] = compact_ids[clusters[v]];
}
}  // namespace

template <typename Types>
struct SclpWorkspace<Types>::Impl {
    using VertexT = typename Types::VertexT;
    using WeightT = typename Types::WeightT;

    thrust::device_vector<VertexT> clusters;
    thrust::device_vector<std::int32_t> low_flags;
    thrust::device_vector<std::int32_t> medium_flags;
    thrust::device_vector<std::int32_t> high_flags;
    thrust::device_vector<VertexT> low_vertices;
    thrust::device_vector<VertexT> medium_vertices;
    thrust::device_vector<VertexT> high_vertices;
    thrust::device_vector<VertexT> proposals;
    thrust::device_vector<VertexT> order;
    thrust::device_vector<VertexT> ordered_targets;
    thrust::device_vector<VertexT> favorites;
    thrust::device_vector<std::int32_t> pair_flags;
    thrust::device_vector<std::int32_t> pair_positions;
    thrust::device_vector<std::int32_t> nonempty_flags;
    thrust::device_vector<std::int32_t> compact_ids;

    thrust::device_vector<std::int64_t> high_edge_counts;
    thrust::device_vector<std::int64_t> high_edge_offsets;

    thrust::device_vector<std::uint32_t> proposal_ties;
    thrust::device_vector<std::uint8_t> radix_temp;
    thrust::device_vector<SclpAdmissionKey> admission_keys;
    thrust::device_vector<SclpAdmissionKey> compact_admission_keys;
    thrust::device_vector<SclpAdmissionKey> sorted_admission_keys;

    thrust::device_vector<unsigned long long> cluster_weights;
    thrust::device_vector<std::uint64_t> affinity_keys;
    thrust::device_vector<std::uint64_t> affinity_values;
    thrust::device_vector<std::uint64_t> unique_keys;
    thrust::device_vector<std::uint64_t> unique_values;
    thrust::device_vector<unsigned long long> best_affinity;
    thrust::device_vector<unsigned long long> unrestricted_best_affinity;
    thrust::device_vector<unsigned long long> current_affinity;
    thrust::device_vector<unsigned long long> best_ties;
    thrust::device_vector<unsigned long long> gains;
    thrust::device_vector<std::uint64_t> ordered_weights;
    thrust::device_vector<std::uint64_t> prefix_weights;
    thrust::device_vector<unsigned long long> base_cluster_weights;
    thrust::device_vector<std::uint64_t> pair_ones;
    thrust::device_vector<std::uint64_t> pair_ranks;
    thrust::device_vector<unsigned long long> counters;
    thrust::device_vector<unsigned long long> role_diagnostics;
    thrust::device_vector<unsigned long long> cut_counter;
    thrust::device_vector<unsigned long long> cross_counter;
    thrust::device_vector<std::uint64_t> diagnostic_member_counts;
    thrust::device_vector<std::uint64_t> diagnostic_unit_weights;
    thrust::device_vector<WeightT> diagnostic_sorted_cluster_weights;
};

template <typename Types>
SclpWorkspace<Types>::SclpWorkspace() : impl(std::make_unique<Impl>()) {}
template <typename Types>
SclpWorkspace<Types>::~SclpWorkspace() = default;
template <typename Types>
SclpWorkspace<Types>::SclpWorkspace(SclpWorkspace&&) noexcept = default;
template <typename Types>
SclpWorkspace<Types>& SclpWorkspace<Types>::operator=(SclpWorkspace&&) noexcept = default;

template <typename Types>
void SclpWorkspace<Types>::reserve_contract_buffers(std::int64_t max_edges) {
    if (max_edges < 0) {
        throw std::invalid_argument("negative contraction buffer size");
    }
    auto& ws = *impl;
    const auto capacity = static_cast<std::size_t>(max_edges);
    ws.affinity_keys.reserve(capacity);
    ws.affinity_values.reserve(capacity);
    ws.unique_keys.reserve(capacity);
    ws.unique_values.reserve(capacity);
    ws.affinity_keys.resize(capacity);
    ws.affinity_values.resize(capacity);
    ws.unique_keys.resize(capacity);
    ws.unique_values.resize(capacity);
    std::size_t radix_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, radix_bytes,
        thrust::raw_pointer_cast(ws.affinity_keys.data()),
        thrust::raw_pointer_cast(ws.unique_keys.data()),
        thrust::raw_pointer_cast(ws.affinity_values.data()),
        thrust::raw_pointer_cast(ws.unique_values.data()),
        static_cast<typename Types::OffsetT>(max_edges)));
    ws.radix_temp.resize(radix_bytes);
}

template <typename Types>
DeviceAggregateResult<Types> aggregate(
    const DeviceWeightedGraph<Types>& graph, SclpWorkspace<Types>& workspace,
    int parts, std::uint32_t seed,
    int level, SclpStats& stats, bool diagnostics) {
    auto& ws = *workspace.impl;
    GpuEventTimer aggregate_timer;
    const auto n = graph.vertices();
    const auto m = graph.edges();
    if (n <= 0 || n > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("SCLP coarsening received an unsupported graph size");
    }
    constexpr int maximum_rounds = 4;
    constexpr bool two_hop_enabled = true;
    constexpr double two_hop_threshold = 0.60;
    const auto total_weight = thrust::reduce(
        graph.vertex_weights.begin(), graph.vertex_weights.end(),
        std::uint64_t{0}, thrust::plus<std::uint64_t>());
    const auto maximum_vertex = thrust::reduce(
        graph.vertex_weights.begin(), graph.vertex_weights.end(),
        std::uint64_t{0}, thrust::maximum<std::uint64_t>());
    const auto denominator = static_cast<std::uint64_t>(kBeta) *
                             static_cast<std::uint64_t>(parts);
    const auto average_cap = total_weight / denominator +
                             (total_weight % denominator != 0 ? 1 : 0);
    stats.capacity = std::max(maximum_vertex, average_cap);

    const int vertex_blocks = static_cast<int>((n + 255) / 256);
    auto& clusters = ws.clusters;
    clusters.resize(static_cast<std::size_t>(n));
    thrust::sequence(clusters.begin(), clusters.end());
    auto& cluster_weights = ws.cluster_weights;
    cluster_weights.assign(
        graph.vertex_weights.begin(), graph.vertex_weights.end());

    // Degree classes are fixed for this graph level and built once. Low and
    // medium vertices are currently warp-per-vertex; only high vertices use
    // the global sort/reduce fallback.
    auto& low_flags = ws.low_flags;
    auto& medium_flags = ws.medium_flags;
    auto& high_flags = ws.high_flags;
    auto& low_vertices = ws.low_vertices;
    auto& medium_vertices = ws.medium_vertices;
    auto& high_vertices = ws.high_vertices;
    low_flags.resize(static_cast<std::size_t>(n));
    medium_flags.resize(static_cast<std::size_t>(n));
    high_flags.resize(static_cast<std::size_t>(n));
    low_vertices.resize(static_cast<std::size_t>(n));
    medium_vertices.resize(static_cast<std::size_t>(n));
    high_vertices.resize(static_cast<std::size_t>(n));
    sclp_degree_class_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.offsets.data()),
        thrust::raw_pointer_cast(low_flags.data()),
        thrust::raw_pointer_cast(medium_flags.data()),
        thrust::raw_pointer_cast(high_flags.data()));
    CUDA_CHECK(cudaGetLastError());
    const auto compact_class = [&](const thrust::device_vector<std::int32_t>& flags,
                                   thrust::device_vector<std::int32_t>& vertices) {
        const auto begin = thrust::make_counting_iterator<std::int32_t>(0);
        const auto end = thrust::copy_if(
            thrust::device, begin, begin + n, flags.begin(), vertices.begin(),
            SclpNonzeroWeight{});
        return static_cast<std::int64_t>(end - vertices.begin());
    };
    const auto low_count = compact_class(low_flags, low_vertices);
    const auto medium_count = compact_class(medium_flags, medium_vertices);
    const auto high_count = compact_class(high_flags, high_vertices);

    auto& high_edge_counts = ws.high_edge_counts;
    auto& high_edge_offsets = ws.high_edge_offsets;
    high_edge_counts.resize(static_cast<std::size_t>(high_count));
    high_edge_offsets.resize(static_cast<std::size_t>(high_count));
    if (high_count > 0) {
        const int blocks = static_cast<int>((high_count + 255) / 256);
        sclp_high_edge_counts_kernel<<<blocks, 256>>>(
            high_count, thrust::raw_pointer_cast(high_vertices.data()),
            thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(high_edge_counts.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::exclusive_scan(
            high_edge_counts.begin(), high_edge_counts.end(),
            high_edge_offsets.begin());
    }
    const auto high_edge_count = high_count == 0 ? std::int64_t{0} :
        thrust::reduce(
            high_edge_counts.begin(), high_edge_counts.end(), std::int64_t{0},
            thrust::plus<std::int64_t>());
    auto& affinity_keys = ws.affinity_keys;
    auto& affinity_values = ws.affinity_values;
    auto& unique_keys = ws.unique_keys;
    auto& unique_values = ws.unique_values;
    if (affinity_keys.size() < static_cast<std::size_t>(high_edge_count)) {
        affinity_keys.resize(static_cast<std::size_t>(high_edge_count));
        affinity_values.resize(static_cast<std::size_t>(high_edge_count));
        unique_keys.resize(static_cast<std::size_t>(high_edge_count));
        unique_values.resize(static_cast<std::size_t>(high_edge_count));
    }
    auto& best_affinity = ws.best_affinity;
    auto& unrestricted_best_affinity = ws.unrestricted_best_affinity;
    auto& current_affinity = ws.current_affinity;
    auto& best_ties = ws.best_ties;
    auto& gains = ws.gains;
    auto& proposal_ties = ws.proposal_ties;
    auto& proposals = ws.proposals;
    auto& order = ws.order;
    auto& ordered_targets = ws.ordered_targets;
    auto& ordered_weights = ws.ordered_weights;
    auto& admission_keys = ws.admission_keys;
    auto& compact_admission_keys = ws.compact_admission_keys;
    auto& sorted_admission_keys = ws.sorted_admission_keys;
    best_affinity.resize(static_cast<std::size_t>(n));
    unrestricted_best_affinity.resize(static_cast<std::size_t>(n));
    current_affinity.resize(static_cast<std::size_t>(n));
    best_ties.resize(static_cast<std::size_t>(n));
    gains.resize(static_cast<std::size_t>(n));
    proposal_ties.resize(static_cast<std::size_t>(n));
    proposals.resize(static_cast<std::size_t>(n));
    order.resize(static_cast<std::size_t>(n));
    ordered_targets.resize(static_cast<std::size_t>(n));
    ordered_weights.resize(static_cast<std::size_t>(n));
    admission_keys.resize(static_cast<std::size_t>(n));
    compact_admission_keys.resize(static_cast<std::size_t>(n));
    sorted_admission_keys.resize(static_cast<std::size_t>(n));
    std::size_t radix_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
        nullptr, radix_temp_bytes,
        thrust::raw_pointer_cast(compact_admission_keys.data()),
        thrust::raw_pointer_cast(sorted_admission_keys.data()),
        static_cast<int>(n), SclpAdmissionKeyDecomposer{}));
    auto& radix_temp = ws.radix_temp;
    auto& prefix_weights = ws.prefix_weights;
    auto& base_cluster_weights = ws.base_cluster_weights;
    auto& counters = ws.counters;
    auto& role_diagnostics = ws.role_diagnostics;
    auto& cut_counter = ws.cut_counter;
    radix_temp.resize(radix_temp_bytes);
    prefix_weights.resize(static_cast<std::size_t>(n));
    base_cluster_weights.resize(static_cast<std::size_t>(n));
    counters.resize(diagnostics ? 3 : 0);
    role_diagnostics.resize(diagnostics ? 3 : 0);
    cut_counter.resize(diagnostics ? 1 : 0);
    const auto build_admission_order = [n, vertex_blocks, &graph, &order,
                                        &ordered_targets, &ordered_weights,
                                        &admission_keys, &compact_admission_keys,
                                        &sorted_admission_keys, &radix_temp,
                                        radix_temp_bytes](
        const thrust::device_vector<std::int32_t>& source_targets,
        const thrust::device_vector<unsigned long long>& source_gains,
        const thrust::device_vector<std::uint32_t>& source_ties) {
        sclp_admission_keys_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(source_targets.data()),
            thrust::raw_pointer_cast(source_gains.data()),
            thrust::raw_pointer_cast(source_ties.data()),
            thrust::raw_pointer_cast(admission_keys.data()));
        CUDA_CHECK(cudaGetLastError());
        const auto end = thrust::copy_if(
            thrust::device, admission_keys.begin(), admission_keys.end(),
            compact_admission_keys.begin(), SclpValidAdmissionKey{});
        const auto count = static_cast<std::int64_t>(
            end - compact_admission_keys.begin());
        if (count == 0) return count;
        auto call_temp_bytes = radix_temp_bytes;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            thrust::raw_pointer_cast(radix_temp.data()), call_temp_bytes,
            thrust::raw_pointer_cast(compact_admission_keys.data()),
            thrust::raw_pointer_cast(sorted_admission_keys.data()),
            static_cast<int>(count), SclpAdmissionKeyDecomposer{}));
        const int blocks = static_cast<int>((count + 255) / 256);
        sclp_decode_admission_keys_kernel<<<blocks, 256>>>(
            count, thrust::raw_pointer_cast(sorted_admission_keys.data()),
            thrust::raw_pointer_cast(graph.vertex_weights.data()),
            thrust::raw_pointer_cast(order.data()),
            thrust::raw_pointer_cast(ordered_targets.data()),
            thrust::raw_pointer_cast(ordered_weights.data()));
        CUDA_CHECK(cudaGetLastError());
        return count;
    };
    const auto device_cluster_cut = [&]() {
        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(cut_counter.data()), 0,
                              sizeof(unsigned long long)));
        weighted_cut_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(graph.edge_weights.data()),
            thrust::raw_pointer_cast(clusters.data()),
            thrust::raw_pointer_cast(cut_counter.data()));
        CUDA_CHECK(cudaGetLastError());
        unsigned long long directed = 0;
        CUDA_CHECK(cudaMemcpy(&directed,
            thrust::raw_pointer_cast(cut_counter.data()), sizeof(directed),
            cudaMemcpyDeviceToHost));
        if (directed & 1ULL) {
            throw std::runtime_error("SCLP cluster cut is not symmetric");
        }
        return static_cast<std::uint64_t>(directed / 2);
    };

    auto form_exact_proposals = [&, level, seed](
        int round, bool filter_roles, std::uint32_t mover_threshold,
        bool measure_affinity) {
        GpuEventTimer affinity_timer;
        thrust::fill(best_affinity.begin(), best_affinity.end(), 0ULL);
        thrust::fill(
            unrestricted_best_affinity.begin(),
            unrestricted_best_affinity.end(), 0ULL);
        thrust::fill(current_affinity.begin(), current_affinity.end(), 0ULL);
        thrust::fill(best_ties.begin(), best_ties.end(), 0ULL);
        const auto role_salt = seed ^
            (static_cast<std::uint32_t>(level) * 0x9e3779b9U) ^
            (static_cast<std::uint32_t>(round) * 0x85ebca6bU) ^ 0x27d4eb2dU;
        const auto tie_salt = seed ^
            (static_cast<std::uint32_t>(level) * 0xc2b2ae35U) ^
            (static_cast<std::uint32_t>(round) * 0x165667b1U) ^ 0xd3a2646cU;
        if (low_count > 0) {
            const int blocks = static_cast<int>(
                (low_count + WARPS_PER_BLOCK * 4 - 1) /
                (WARPS_PER_BLOCK * 4));
            sclp_affinity_low_match_kernel<<<blocks, 256>>>(
                low_count, thrust::raw_pointer_cast(low_vertices.data()),
                thrust::raw_pointer_cast(graph.offsets.data()),
                thrust::raw_pointer_cast(graph.neighbors.data()),
                thrust::raw_pointer_cast(graph.edge_weights.data()),
                thrust::raw_pointer_cast(clusters.data()),
                role_salt, mover_threshold, tie_salt, filter_roles,
                thrust::raw_pointer_cast(current_affinity.data()),
                thrust::raw_pointer_cast(best_affinity.data()),
                thrust::raw_pointer_cast(best_ties.data()),
                thrust::raw_pointer_cast(unrestricted_best_affinity.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        if (medium_count > 0) {
            const int threads = SCLP_AFFINITY_WARPS_PER_BLOCK * WARP_SIZE;
            const int blocks = static_cast<int>(
                (medium_count + SCLP_AFFINITY_WARPS_PER_BLOCK - 1) /
                SCLP_AFFINITY_WARPS_PER_BLOCK);
            constexpr std::size_t shared_bytes =
                SCLP_AFFINITY_WARPS_PER_BLOCK * SCLP_MEDIUM_HASH_SIZE *
                (sizeof(std::int32_t) + sizeof(unsigned long long));
            sclp_affinity_medium_hash_kernel<<<blocks, threads, shared_bytes>>>(
                medium_count, thrust::raw_pointer_cast(medium_vertices.data()),
                thrust::raw_pointer_cast(graph.offsets.data()),
                thrust::raw_pointer_cast(graph.neighbors.data()),
                thrust::raw_pointer_cast(graph.edge_weights.data()),
                thrust::raw_pointer_cast(clusters.data()),
                role_salt, mover_threshold, tie_salt, filter_roles,
                thrust::raw_pointer_cast(current_affinity.data()),
                thrust::raw_pointer_cast(best_affinity.data()),
                thrust::raw_pointer_cast(best_ties.data()),
                thrust::raw_pointer_cast(unrestricted_best_affinity.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        std::int64_t unique_count = 0;
        if (high_edge_count > 0) {
            sclp_fill_affinity_high_kernel<<<static_cast<int>(high_count), 256>>>(
                high_count, thrust::raw_pointer_cast(high_vertices.data()),
                thrust::raw_pointer_cast(high_edge_offsets.data()),
                thrust::raw_pointer_cast(graph.offsets.data()),
                thrust::raw_pointer_cast(graph.neighbors.data()),
                thrust::raw_pointer_cast(graph.edge_weights.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(affinity_keys.data()),
                thrust::raw_pointer_cast(affinity_values.data()));
            CUDA_CHECK(cudaGetLastError());
            std::size_t affinity_radix_bytes = 0;
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                nullptr, affinity_radix_bytes,
                thrust::raw_pointer_cast(affinity_keys.data()),
                thrust::raw_pointer_cast(unique_keys.data()),
                thrust::raw_pointer_cast(affinity_values.data()),
                thrust::raw_pointer_cast(unique_values.data()),
                static_cast<typename Types::OffsetT>(high_edge_count)));
            radix_temp.resize(std::max(radix_temp.size(), affinity_radix_bytes));
            auto affinity_radix_call_bytes = affinity_radix_bytes;
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                thrust::raw_pointer_cast(radix_temp.data()),
                affinity_radix_call_bytes,
                thrust::raw_pointer_cast(affinity_keys.data()),
                thrust::raw_pointer_cast(unique_keys.data()),
                thrust::raw_pointer_cast(affinity_values.data()),
                thrust::raw_pointer_cast(unique_values.data()),
                static_cast<typename Types::OffsetT>(high_edge_count)));
            const auto valid_end = thrust::lower_bound(
                thrust::device, unique_keys.begin(),
                unique_keys.begin() + high_edge_count,
                INVALID_KEY);
            const auto reduced = thrust::reduce_by_key(
                thrust::device, unique_keys.begin(), valid_end,
                unique_values.begin(), affinity_keys.begin(), affinity_values.begin());
            unique_count = static_cast<std::int64_t>(
                reduced.first - affinity_keys.begin());
        }
        if (unique_count > 0) {
            const int blocks = static_cast<int>((unique_count + 255) / 256);
            sclp_affinity_baseline_kernel<<<blocks, 256>>>(
                unique_count, thrust::raw_pointer_cast(affinity_keys.data()),
                thrust::raw_pointer_cast(affinity_values.data()),
                thrust::raw_pointer_cast(clusters.data()),
                role_salt, mover_threshold, filter_roles,
                thrust::raw_pointer_cast(current_affinity.data()),
                thrust::raw_pointer_cast(best_affinity.data()),
                thrust::raw_pointer_cast(unrestricted_best_affinity.data()));
            CUDA_CHECK(cudaGetLastError());
            sclp_best_tie_kernel<<<blocks, 256>>>(
                unique_count, thrust::raw_pointer_cast(affinity_keys.data()),
                thrust::raw_pointer_cast(affinity_values.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(best_affinity.data()),
                tie_salt, role_salt, mover_threshold, filter_roles,
                thrust::raw_pointer_cast(best_ties.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        if (diagnostics) {
            CUDA_CHECK(cudaMemset(
                thrust::raw_pointer_cast(role_diagnostics.data()), 0,
                3 * sizeof(unsigned long long)));
        }
        sclp_decode_gain_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(clusters.data()),
            thrust::raw_pointer_cast(current_affinity.data()),
            thrust::raw_pointer_cast(best_affinity.data()),
            thrust::raw_pointer_cast(best_ties.data()),
            thrust::raw_pointer_cast(unrestricted_best_affinity.data()),
            role_salt, mover_threshold, filter_roles,
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(gains.data()),
            thrust::raw_pointer_cast(proposal_ties.data()),
            diagnostics ? thrust::raw_pointer_cast(role_diagnostics.data()) : nullptr);
        CUDA_CHECK(cudaGetLastError());
        const auto affinity_seconds = affinity_timer.seconds();
        if (measure_affinity) stats.affinity_seconds += affinity_seconds;
    };

    std::int64_t cluster_count = n;
    const std::int64_t desired_clusters = (n + 1) / 2;
    for (int round = 0; round < maximum_rounds; ++round) {
        if (static_cast<double>(cluster_count) <= 0.60 * n) break;
        const auto removable_clusters = std::max<std::int64_t>(
            0, cluster_count - desired_clusters);
        const double mover_fraction = std::min(
            0.50, static_cast<double>(removable_clusters) /
                  static_cast<double>(cluster_count));
        const auto mover_threshold = static_cast<std::uint32_t>(
            mover_fraction * 4294967296.0);
        form_exact_proposals(round, true, mover_threshold, true);
        if (diagnostics) {
            unsigned long long host_role_diagnostics[3] = {};
            CUDA_CHECK(cudaMemcpy(
                host_role_diagnostics,
                thrust::raw_pointer_cast(role_diagnostics.data()),
                sizeof(host_role_diagnostics), cudaMemcpyDeviceToHost));
            stats.positive_gain_vertices = host_role_diagnostics[0];
            stats.role_blocked_vertices = host_role_diagnostics[1];
            stats.role_blocked_gain = host_role_diagnostics[2];
        }
        GpuEventTimer admission_timer;
        const auto cut_before = diagnostics ? device_cluster_cut() : 0;
        const auto valid_count = build_admission_order(
            proposals, gains, proposal_ties);
        unsigned long long accepted = 0;
        unsigned long long rejected = 0;
        unsigned long long predicted_gain = 0;
        if (valid_count > 0) {
            const int blocks = static_cast<int>((valid_count + 255) / 256);
            thrust::inclusive_scan_by_key(
                thrust::device, ordered_targets.begin(),
                ordered_targets.begin() + valid_count, ordered_weights.begin(),
                prefix_weights.begin());
            thrust::copy(cluster_weights.begin(), cluster_weights.end(),
                         base_cluster_weights.begin());
            if (diagnostics) {
                CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counters.data()), 0,
                                      3 * sizeof(unsigned long long)));
            }
            auto* admission_counters = diagnostics
                ? thrust::raw_pointer_cast(counters.data()) : nullptr;
            sclp_commit_kernel<<<blocks, 256>>>(
                valid_count, thrust::raw_pointer_cast(order.data()),
                thrust::raw_pointer_cast(ordered_targets.data()),
                thrust::raw_pointer_cast(ordered_weights.data()),
                thrust::raw_pointer_cast(prefix_weights.data()),
                thrust::raw_pointer_cast(base_cluster_weights.data()),
                stats.capacity, thrust::raw_pointer_cast(gains.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(cluster_weights.data()),
                admission_counters,
                diagnostics ? admission_counters + 1 : nullptr,
                diagnostics ? admission_counters + 2 : nullptr);
            CUDA_CHECK(cudaGetLastError());
            if (diagnostics) {
                unsigned long long host_counters[3] = {};
                CUDA_CHECK(cudaMemcpy(
                    host_counters, admission_counters, sizeof(host_counters),
                    cudaMemcpyDeviceToHost));
                accepted = host_counters[0];
                rejected = host_counters[1];
                predicted_gain = host_counters[2];
            }
        }

        cluster_count = static_cast<std::int64_t>(thrust::count_if(
            cluster_weights.begin(), cluster_weights.end(), SclpNonzeroWeight{}));
        stats.lp_accepted += accepted;
        stats.capacity_rejected += rejected;
        stats.proposal_count += static_cast<std::uint64_t>(valid_count);
        ++stats.rounds;
        stats.admission_seconds += admission_timer.seconds();
        const auto cut_after = diagnostics ? device_cluster_cut() : 0;
        if (diagnostics && cut_after > cut_before) {
            throw std::runtime_error("SCLP synchronous batch increased cluster cut");
        }
        std::cout << "ml_gpu_sclp_round level=" << level
                  << " round=" << round
                  << " proposals=" << valid_count;
        if (diagnostics) {
            std::cout << " accepted=" << accepted
                      << " capacity_rejected=" << rejected
                      << " predicted_gain=" << predicted_gain
                      << " capacity_rejection_ratio="
                      << (valid_count == 0 ? 0.0 :
                          static_cast<double>(rejected) /
                          static_cast<double>(valid_count))
                      << " cut_before=" << cut_before
                      << " cut_after=" << cut_after
                      << " actual_gain=" << (cut_before - cut_after)
                      << " positive_gain_vertices="
                      << stats.positive_gain_vertices
                      << " role_blocked=" << stats.role_blocked_vertices
                      << " role_blocked_gain=" << stats.role_blocked_gain;
        }
        std::cout << " mover_fraction=" << mover_fraction
                  << " clusters=" << cluster_count
                  << " contraction_ratio="
                  << static_cast<double>(cluster_count) / static_cast<double>(n)
                  << '\n';
        if (static_cast<double>(cluster_count) <= 0.60 * n) break;
    }

    // Simple two-hop fallback: remaining singleton vertices with the same
    // one-hop favorite form at most one deterministic capacity-bounded group.
    // This is a fallback, not part of the SCLP quality claim.
    const bool two_hop_triggered = two_hop_enabled &&
        static_cast<double>(cluster_count) > two_hop_threshold * n;
    if (two_hop_triggered || diagnostics) {
        GpuEventTimer two_hop_timer;
        const auto fallback_cut_before = diagnostics ? device_cluster_cut() : 0;
        form_exact_proposals(maximum_rounds, false, 0, false);
        auto& favorites = ws.favorites;
        favorites.resize(static_cast<std::size_t>(n));
        sclp_singleton_favorite_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(graph.vertex_weights.data()),
            thrust::raw_pointer_cast(cluster_weights.data()),
            thrust::raw_pointer_cast(clusters.data()),
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(favorites.data()));
        CUDA_CHECK(cudaGetLastError());
        const auto singleton_candidates = build_admission_order(
            favorites, gains, proposal_ties);
        stats.singleton_with_favorite = static_cast<std::uint64_t>(
            singleton_candidates);
        if (singleton_candidates > 0) {
            const int blocks = static_cast<int>((singleton_candidates + 255) / 256);
            auto& pair_ones = ws.pair_ones;
            auto& pair_ranks = ws.pair_ranks;
            pair_ones.resize(static_cast<std::size_t>(singleton_candidates));
            pair_ranks.resize(static_cast<std::size_t>(singleton_candidates));
            thrust::fill(pair_ones.begin(), pair_ones.end(), std::uint64_t{1});
            thrust::inclusive_scan_by_key(
                thrust::device, ordered_targets.begin(),
                ordered_targets.begin() + singleton_candidates,
                pair_ones.begin(), pair_ranks.begin());
            auto& pair_flags = ws.pair_flags;
            auto& pair_positions = ws.pair_positions;
            pair_flags.resize(static_cast<std::size_t>(singleton_candidates));
            pair_positions.resize(static_cast<std::size_t>(singleton_candidates));
            sclp_pair_flags_kernel<<<blocks, 256>>>(
                singleton_candidates, thrust::raw_pointer_cast(order.data()),
                thrust::raw_pointer_cast(ordered_weights.data()),
                thrust::raw_pointer_cast(pair_ranks.data()), stats.capacity,
                thrust::raw_pointer_cast(pair_flags.data()));
            CUDA_CHECK(cudaGetLastError());
            thrust::exclusive_scan(
                pair_flags.begin(), pair_flags.end(), pair_positions.begin());
            const auto pair_count = static_cast<std::uint64_t>(thrust::count(
                pair_flags.begin(), pair_flags.end(), std::int32_t{1}));
            stats.pairable_singletons = 2 * pair_count;
            const auto merge_budget = static_cast<std::uint64_t>(
                std::max<std::int64_t>(0, cluster_count - desired_clusters));
            const auto selected_count = std::min(pair_count, merge_budget);
            if (two_hop_triggered && selected_count > 0) {
                auto* merge_counter = diagnostics
                    ? thrust::raw_pointer_cast(counters.data()) : nullptr;
                if (diagnostics) {
                    CUDA_CHECK(cudaMemset(
                        merge_counter, 0, sizeof(unsigned long long)));
                }
                sclp_two_hop_pair_commit_kernel<<<blocks, 256>>>(
                    singleton_candidates, thrust::raw_pointer_cast(order.data()),
                    thrust::raw_pointer_cast(pair_flags.data()),
                    thrust::raw_pointer_cast(pair_positions.data()), merge_budget,
                    thrust::raw_pointer_cast(ordered_weights.data()),
                    thrust::raw_pointer_cast(clusters.data()),
                    thrust::raw_pointer_cast(cluster_weights.data()),
                    merge_counter);
                CUDA_CHECK(cudaGetLastError());
                stats.two_hop_merged = selected_count;
                if (diagnostics) {
                    CUDA_CHECK(cudaMemcpy(
                        &stats.two_hop_merged, merge_counter,
                        sizeof(stats.two_hop_merged), cudaMemcpyDeviceToHost));
                }
            }
        }
        cluster_count = static_cast<std::int64_t>(thrust::count_if(
            cluster_weights.begin(), cluster_weights.end(), SclpNonzeroWeight{}));
        if (diagnostics) {
            const auto fallback_cut_after = device_cluster_cut();
            if (fallback_cut_after > fallback_cut_before) {
                throw std::runtime_error("SCLP two-hop pairing increased cluster cut");
            }
            std::cout << "ml_gpu_sclp_two_hop level=" << level
                      << " enabled=" << (two_hop_enabled ? 1 : 0)
                      << " threshold=" << two_hop_threshold
                      << " triggered=" << (two_hop_triggered ? 1 : 0)
                      << " singleton_with_favorite="
                      << stats.singleton_with_favorite
                      << " pairable_singletons=" << stats.pairable_singletons
                      << " merged=" << stats.two_hop_merged
                      << " cut_before=" << fallback_cut_before
                      << " cut_after=" << fallback_cut_after
                      << " actual_gain="
                      << (fallback_cut_before - fallback_cut_after) << '\n';
        }
        stats.two_hop_seconds = two_hop_timer.seconds();
    }

    GpuEventTimer compact_timer;
    auto& nonempty_flags = ws.nonempty_flags;
    auto& compact_ids = ws.compact_ids;
    nonempty_flags.resize(static_cast<std::size_t>(n));
    compact_ids.resize(static_cast<std::size_t>(n));
    sclp_nonempty_cluster_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(cluster_weights.data()),
        thrust::raw_pointer_cast(nonempty_flags.data()));
    CUDA_CHECK(cudaGetLastError());
    thrust::exclusive_scan(nonempty_flags.begin(), nonempty_flags.end(),
                           compact_ids.begin());

    DeviceAggregateResult<Types> out;
    out.coarse_vertices = static_cast<std::int32_t>(cluster_count);
    out.capacity = stats.capacity;
    out.map.resize(static_cast<std::size_t>(n));
    sclp_compact_map_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(clusters.data()),
        thrust::raw_pointer_cast(compact_ids.data()),
        thrust::raw_pointer_cast(out.map.data()));
    CUDA_CHECK(cudaGetLastError());
    out.vertex_weights.resize(static_cast<std::size_t>(out.coarse_vertices));
    thrust::fill(
        out.vertex_weights.begin(), out.vertex_weights.end(),
        typename Types::WeightT{0});
    coarse_vertex_weights_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.vertex_weights.data()),
        thrust::raw_pointer_cast(out.map.data()),
        thrust::raw_pointer_cast(out.vertex_weights.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    out.maximum_weight = thrust::reduce(
        out.vertex_weights.begin(), out.vertex_weights.end(),
        std::uint64_t{0}, thrust::maximum<std::uint64_t>());
    if (out.maximum_weight > stats.capacity) {
        throw std::runtime_error("SCLP coarsening exceeded cluster capacity");
    }
    stats.compact_seconds = compact_timer.seconds();
    stats.aggregate_seconds = aggregate_timer.seconds();
    std::uint64_t weight_p50 = 0;
    std::uint64_t weight_p90 = 0;
    std::uint64_t weight_p99 = 0;
    if (diagnostics) {
        // A singleton cluster has exactly one member; count it from the compact
        // map rather than comparing vertex and aggregate weights.
        auto& member_counts = ws.diagnostic_member_counts;
        auto& unit_weights = ws.diagnostic_unit_weights;
        member_counts.resize(static_cast<std::size_t>(out.coarse_vertices));
        unit_weights.resize(static_cast<std::size_t>(n));
        thrust::fill(member_counts.begin(), member_counts.end(), std::uint64_t{0});
        thrust::fill(unit_weights.begin(), unit_weights.end(), std::uint64_t{1});
        coarse_vertex_weights_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(unit_weights.data()),
            thrust::raw_pointer_cast(out.map.data()),
            thrust::raw_pointer_cast(member_counts.data()));
        CUDA_CHECK(cudaGetLastError());
        stats.singleton_count = static_cast<std::uint64_t>(thrust::count(
            member_counts.begin(), member_counts.end(), std::uint64_t{1}));
        auto& sorted_cluster_weights = ws.diagnostic_sorted_cluster_weights;
        sorted_cluster_weights = out.vertex_weights;
        thrust::sort(sorted_cluster_weights.begin(), sorted_cluster_weights.end());
        const auto weight_percentile = [&](double fraction) {
            const auto index = std::min<std::size_t>(
                sorted_cluster_weights.size() - 1,
                static_cast<std::size_t>(fraction * static_cast<double>(
                    sorted_cluster_weights.size() - 1)));
            return static_cast<std::uint64_t>(sorted_cluster_weights[index]);
        };
        weight_p50 = weight_percentile(0.50);
        weight_p90 = weight_percentile(0.90);
        weight_p99 = weight_percentile(0.99);
    }
    const double capacity_rejection_ratio = stats.proposal_count == 0 ? 0.0 :
        static_cast<double>(stats.capacity_rejected) /
        static_cast<double>(stats.proposal_count);
    std::cout << "ml_gpu_sclp level=" << level
              << " fine_vertices=" << n
              << " coarse_vertices=" << out.coarse_vertices
              << " contraction_ratio="
              << static_cast<double>(out.coarse_vertices) / static_cast<double>(n)
              << " beta=" << kBeta
              << " cluster_cap=" << stats.capacity
              << " lp_rounds=" << stats.rounds
              << " two_hop_enabled=" << (two_hop_enabled ? 1 : 0)
              << " two_hop_threshold=" << two_hop_threshold;
    if (diagnostics) {
        std::cout << " lp_accepted=" << stats.lp_accepted
              << " capacity_rejected=" << stats.capacity_rejected
              << " capacity_rejection_ratio=" << capacity_rejection_ratio
              << " singleton=" << stats.singleton_count
              << " positive_gain_vertices=" << stats.positive_gain_vertices
              << " role_blocked=" << stats.role_blocked_vertices
              << " role_blocked_gain=" << stats.role_blocked_gain
              << " singleton_with_favorite=" << stats.singleton_with_favorite
              << " pairable_singletons=" << stats.pairable_singletons
              << " two_hop_merged=" << stats.two_hop_merged
              << " weight_p50=" << weight_p50
              << " weight_p90=" << weight_p90
              << " weight_p99=" << weight_p99
              << " affinity_seconds=" << stats.affinity_seconds
              << " admission_seconds=" << stats.admission_seconds
              << " two_hop_seconds=" << stats.two_hop_seconds;
    }
    std::cout
              << " weight_max=" << out.maximum_weight
              << " weight_max_over_cap="
              << static_cast<double>(out.maximum_weight) /
                 static_cast<double>(stats.capacity)
              << " low_vertices=" << low_count
              << " medium_vertices=" << medium_count
              << " high_vertices=" << high_count
              << " hub_edge_entries=" << high_edge_count
              << " hub_sort_fraction="
              << (m == 0 ? 0.0 :
                  static_cast<double>(high_edge_count) / static_cast<double>(m))
              << " seed=" << seed
              << " diagnostics=" << (diagnostics ? 1 : 0) << '\n';
    return out;
}

template <typename Types>
DeviceWeightedGraph<Types> contract(
    DeviceWeightedGraph<Types> fine, DeviceAggregateResult<Types>&& aggregate,
    SclpWorkspace<Types>& workspace, ContractionTimings& timings) {
    GpuEventTimer contraction_timer;
    auto& ws = *workspace.impl;
    const auto n = fine.vertices();
    const auto m = fine.edges();
    const int warp_blocks = static_cast<int>(
        (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    auto& keys = ws.affinity_keys;
    auto& values = ws.affinity_values;
    auto& unique_keys = ws.unique_keys;
    auto& unique_values = ws.unique_values;
    auto& cross_counter = ws.cross_counter;
    if (keys.size() < static_cast<std::size_t>(m)) {
        keys.resize(static_cast<std::size_t>(m));
        values.resize(static_cast<std::size_t>(m));
    }
    cross_counter.resize(1);
    GpuEventTimer compact_timer;
    thrust::fill(cross_counter.begin(), cross_counter.end(), 0ULL);
    compact_cross_edges_warp_kernel<<<warp_blocks, 256>>>(
        n, thrust::raw_pointer_cast(fine.offsets.data()),
        thrust::raw_pointer_cast(fine.neighbors.data()),
        thrust::raw_pointer_cast(fine.edge_weights.data()),
        thrust::raw_pointer_cast(aggregate.map.data()),
        thrust::raw_pointer_cast(keys.data()),
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(cross_counter.data()));
    CUDA_CHECK(cudaGetLastError());
    unsigned long long cross_edges = 0;
    CUDA_CHECK(cudaMemcpy(
        &cross_edges, thrust::raw_pointer_cast(cross_counter.data()),
        sizeof(cross_edges), cudaMemcpyDeviceToHost));
    timings.compact_seconds = compact_timer.seconds();
    if (unique_keys.size() < static_cast<std::size_t>(cross_edges)) {
        unique_keys.resize(static_cast<std::size_t>(cross_edges));
        unique_values.resize(static_cast<std::size_t>(cross_edges));
    }
    std::size_t contraction_radix_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, contraction_radix_bytes,
        thrust::raw_pointer_cast(keys.data()),
        thrust::raw_pointer_cast(unique_keys.data()),
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(unique_values.data()),
        static_cast<typename Types::OffsetT>(cross_edges)));
    ws.radix_temp.resize(contraction_radix_bytes);
    GpuEventTimer sort_timer;
    auto call_radix_bytes = contraction_radix_bytes;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        thrust::raw_pointer_cast(ws.radix_temp.data()), call_radix_bytes,
        thrust::raw_pointer_cast(keys.data()),
        thrust::raw_pointer_cast(unique_keys.data()),
        thrust::raw_pointer_cast(values.data()),
        thrust::raw_pointer_cast(unique_values.data()),
        static_cast<typename Types::OffsetT>(cross_edges)));
    timings.sort_seconds = sort_timer.seconds();
    GpuEventTimer reduce_timer;
    const auto reduced = thrust::reduce_by_key(
        thrust::device, unique_keys.begin(),
        unique_keys.begin() + cross_edges,
        unique_values.begin(), keys.begin(), values.begin());
    timings.reduce_seconds = reduce_timer.seconds();
    const auto coarse_edges = static_cast<std::int64_t>(reduced.first - keys.begin());
    std::cout << "ml_gpu_contract_cross_edge_entries=" << cross_edges
              << " input_edge_entries=" << m
              << " sort_fraction="
              << (m == 0 ? 0.0 :
                  static_cast<double>(cross_edges) / static_cast<double>(m))
              << '\n';

    DeviceWeightedGraph<Types> coarse = std::move(fine);
    GpuEventTimer vertex_weight_timer;
    coarse.vertex_weights.resize(static_cast<std::size_t>(aggregate.coarse_vertices));
    thrust::copy(
        aggregate.vertex_weights.begin(), aggregate.vertex_weights.end(),
        coarse.vertex_weights.begin());
    timings.vertex_weight_seconds = vertex_weight_timer.seconds();
    GpuEventTimer csr_build_timer;
    coarse.offsets.resize(static_cast<std::size_t>(aggregate.coarse_vertices) + 1);
    thrust::fill(
        coarse.offsets.begin(), coarse.offsets.end(), coarse_edges);
    coarse.neighbors.resize(static_cast<std::size_t>(coarse_edges));
    coarse.edge_weights.resize(static_cast<std::size_t>(coarse_edges));
    if (coarse_edges > 0) {
        const int blocks = static_cast<int>((coarse_edges + 255) / 256);
        write_coarse_csr_kernel<<<blocks, 256>>>(
            coarse_edges, thrust::raw_pointer_cast(keys.data()),
            thrust::raw_pointer_cast(values.data()),
            thrust::raw_pointer_cast(coarse.neighbors.data()),
            thrust::raw_pointer_cast(coarse.edge_weights.data()),
            thrust::raw_pointer_cast(coarse.offsets.data()));
        CUDA_CHECK(cudaGetLastError());
    }
    thrust::inclusive_scan(
        thrust::device, coarse.offsets.rbegin(), coarse.offsets.rend(),
        coarse.offsets.rbegin(),
        thrust::minimum<typename Types::OffsetT>());
    timings.csr_build_seconds = csr_build_timer.seconds();
    timings.total_seconds = contraction_timer.seconds();
    return coarse;
}

namespace {

template <typename Types>
void validate_coarsening_step(
    const WeightedGraph<Types>& fine, const WeightedGraph<Types>& coarse,
    const std::vector<typename Types::VertexT>& map,
    std::uint64_t cluster_cap, int level, bool strict_verify) {
    using VertexT = typename Types::VertexT;
    if (map.size() != static_cast<std::size_t>(fine.vertices())) {
        throw std::runtime_error("coarsening map has the wrong length");
    }
    if (coarse.vertices() <= 0 || coarse.vertices() > fine.vertices()) {
        throw std::runtime_error("coarsening produced an invalid vertex count");
    }
    std::vector<std::uint64_t> recomputed(
        static_cast<std::size_t>(coarse.vertices()), 0);
    std::vector<std::uint64_t> coverage(
        static_cast<std::size_t>(coarse.vertices()), 0);
    for (std::int64_t v = 0; v < fine.vertices(); ++v) {
        const auto c = map[static_cast<std::size_t>(v)];
        if (c < 0 || c >= coarse.vertices()) {
            throw std::runtime_error("coarsening map contains an out-of-range id");
        }
        recomputed[static_cast<std::size_t>(c)] += static_cast<std::uint64_t>(
            fine.vertex_weights[static_cast<std::size_t>(v)]);
        ++coverage[static_cast<std::size_t>(c)];
    }
    const auto sum_weights = [](const auto& weights) {
        return std::accumulate(
            weights.begin(), weights.end(), std::uint64_t{0},
            [](std::uint64_t total, auto value) {
                return total + static_cast<std::uint64_t>(value);
            });
    };
    if (sum_weights(fine.vertex_weights) != sum_weights(coarse.vertex_weights)) {
        throw std::runtime_error("coarsening did not conserve vertex weight");
    }
    for (std::int64_t c = 0; c < coarse.vertices(); ++c) {
        if (coverage[static_cast<std::size_t>(c)] == 0 ||
            recomputed[static_cast<std::size_t>(c)] != static_cast<std::uint64_t>(
                coarse.vertex_weights[static_cast<std::size_t>(c)])) {
            throw std::runtime_error("coarsening cluster weight mismatch");
        }
        if (static_cast<std::uint64_t>(
                coarse.vertex_weights[static_cast<std::size_t>(c)]) > cluster_cap) {
            throw std::runtime_error("coarse point exceeds configured weight cap");
        }
    }
    if (strict_verify) {
        validate_weighted_csr(coarse, false);
    } else {
        validate_weighted_shape(coarse, false);
    }

    std::vector<VertexT> coarse_labels(
        static_cast<std::size_t>(coarse.vertices()));
    for (std::int64_t c = 0; c < coarse.vertices(); ++c) {
        coarse_labels[static_cast<std::size_t>(c)] =
            static_cast<VertexT>((c * 2654435761ULL + 17) % 23);
    }
    std::vector<VertexT> fine_labels(map.size());
    for (std::size_t v = 0; v < map.size(); ++v) {
        fine_labels[v] = coarse_labels[static_cast<std::size_t>(map[v])];
    }
    if (host_cut(fine, fine_labels) != host_cut(coarse, coarse_labels)) {
        throw std::runtime_error("cut is not conserved by coarse projection");
    }
    std::cout << "ml_layer_verify level=" << level
              << " map=ok weights=ok csr=ok projection_cut=ok\n";
}

}  // namespace

template <typename Types>
Hierarchy<Types> coarsen(
    const WeightedGraph<Types>& graph, const CoarsenOptions& options) {
    using VertexT = typename Types::VertexT;
    const bool diagnostics = std::getenv("SCLP_DIAGNOSTICS") != nullptr;
    const bool verify = options.strict_verify;
    const auto host_graph_build_start = std::chrono::steady_clock::now();
    Hierarchy<Types> hierarchy;
    hierarchy.levels.push_back(graph);
    const auto host_graph_build_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - host_graph_build_start).count();

    const auto input_verify_start = std::chrono::steady_clock::now();
    if (options.strict_verify) {
        validate_weighted_csr(hierarchy.levels.front(), true);
    } else {
        validate_weighted_shape(hierarchy.levels.front(), true);
    }
    const auto input_verify_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - input_verify_start).count();
    std::cout << "ml_input_verify_seconds=" << input_verify_seconds
              << " status=ok mode="
              << (options.strict_verify ? "strict" : "fast") << '\n';

    const auto device_input_start = std::chrono::steady_clock::now();
    auto current = make_device_weighted(hierarchy.levels.front());
    SclpWorkspace<Types> workspace;
    workspace.reserve_contract_buffers(current.edges());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto device_input_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - device_input_start).count();
    std::cout << "ml_gpu_device_input_seconds=" << device_input_seconds
              << " method=sclp\n";

    const std::int64_t cutoff = std::max<std::int64_t>(
        32, kBeta * static_cast<std::int64_t>(options.parts));
    const auto core_start = std::chrono::steady_clock::now();
    double device_algorithm_seconds = 0.0;
    double aggregate_wall_seconds = 0.0;
    double contraction_wall_seconds = 0.0;
    double snapshot_seconds = 0.0;
    double aggregate_gpu_seconds = 0.0;
    double affinity_gpu_seconds = 0.0;
    double admission_gpu_seconds = 0.0;
    double two_hop_gpu_seconds = 0.0;
    double compact_gpu_seconds = 0.0;
    double contraction_gpu_seconds = 0.0;
    double contraction_compact_gpu_seconds = 0.0;
    double contraction_sort_gpu_seconds = 0.0;
    double contraction_reduce_gpu_seconds = 0.0;
    double contraction_csr_gpu_seconds = 0.0;
    double contraction_vertex_weight_gpu_seconds = 0.0;
    int level = 0;
    for (; level < options.max_levels && current.vertices() > cutoff; ++level) {
        const auto level_seed = options.seed + static_cast<std::uint32_t>(level);
        std::cout << "ml_coarsen_begin level=" << level
                  << " vertices=" << current.vertices()
                  << " edges=" << current.edges()
                  << " method=sclp seed=" << level_seed << '\n';
        SclpStats stats;
        const auto aggregate_start = std::chrono::steady_clock::now();
        auto aggregate_result = aggregate(
            current, workspace, options.parts, level_seed,
            level, stats, diagnostics);
        const auto level_aggregate_wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - aggregate_start).count();
        device_algorithm_seconds += level_aggregate_wall;
        aggregate_wall_seconds += level_aggregate_wall;
        const double contraction =
            static_cast<double>(aggregate_result.coarse_vertices) /
            static_cast<double>(current.vertices());
        affinity_gpu_seconds += stats.affinity_seconds;
        admission_gpu_seconds += stats.admission_seconds;
        two_hop_gpu_seconds += stats.two_hop_seconds;
        compact_gpu_seconds += stats.compact_seconds;
        aggregate_gpu_seconds += stats.aggregate_seconds;
        const auto aggregate_other_seconds = std::max(
            0.0, stats.aggregate_seconds - stats.affinity_seconds -
            stats.admission_seconds - stats.two_hop_seconds -
            stats.compact_seconds);
        std::cout << "ml_coarsen_map level=" << level
                  << " coarse_vertices=" << aggregate_result.coarse_vertices
                  << " ratio=" << contraction
                  << " maximum_weight=" << aggregate_result.maximum_weight
                  << " capacity=" << aggregate_result.capacity << '\n';
        if (aggregate_result.coarse_vertices >= current.vertices() ||
            contraction > options.stop_contraction_ratio) {
            std::cout << "ml_gpu_timing level=" << level
                      << " affinity_seconds=" << stats.affinity_seconds
                      << " admission_seconds=" << stats.admission_seconds
                      << " two_hop_seconds=" << stats.two_hop_seconds
                      << " compact_seconds=" << stats.compact_seconds
                      << " aggregate_other_seconds=" << aggregate_other_seconds
                      << " aggregate_total_seconds=" << stats.aggregate_seconds
                      << " contraction_seconds=0\n";
            std::cout << "ml_coarsen_stop reason=insufficient_contraction\n";
            hierarchy.stop_reason = "insufficient_contraction";
            break;
        }

        ContractionTimings contract_timings;
        const auto contract_wall_start = std::chrono::steady_clock::now();
        auto coarse_device = contract(
            std::move(current), std::move(aggregate_result),
            workspace, contract_timings);
        const auto level_contract_wall = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - contract_wall_start).count();
        device_algorithm_seconds += level_contract_wall;
        contraction_wall_seconds += level_contract_wall;
        contraction_gpu_seconds += contract_timings.total_seconds;
        contraction_compact_gpu_seconds += contract_timings.compact_seconds;
        contraction_sort_gpu_seconds += contract_timings.sort_seconds;
        contraction_reduce_gpu_seconds += contract_timings.reduce_seconds;
        contraction_csr_gpu_seconds += contract_timings.csr_build_seconds;
        contraction_vertex_weight_gpu_seconds +=
            contract_timings.vertex_weight_seconds;
        std::cout << "ml_gpu_timing level=" << level
                  << " affinity_seconds=" << stats.affinity_seconds
                  << " admission_seconds=" << stats.admission_seconds
                  << " two_hop_seconds=" << stats.two_hop_seconds
                  << " compact_seconds=" << stats.compact_seconds
                  << " aggregate_other_seconds=" << aggregate_other_seconds
                  << " aggregate_total_seconds=" << stats.aggregate_seconds
                  << " contraction_seconds=" << contract_timings.total_seconds
                  << '\n';
        std::cout << "ml_gpu_contraction_timing level=" << level
                  << " compact_seconds=" << contract_timings.compact_seconds
                  << " sort_seconds=" << contract_timings.sort_seconds
                  << " reduce_seconds=" << contract_timings.reduce_seconds
                  << " csr_build_seconds=" << contract_timings.csr_build_seconds
                  << " vertex_weight_seconds="
                  << contract_timings.vertex_weight_seconds
                  << " total_seconds=" << contract_timings.total_seconds << '\n';
        std::cout << "ml_gpu_contract_seconds level=" << level
                  << " seconds=" << contract_timings.total_seconds
                  << " coarse_edges=" << coarse_device.edges() << '\n';

        const auto snapshot_start = std::chrono::steady_clock::now();
        std::vector<VertexT> host_map(aggregate_result.map.size());
        thrust::copy(
            aggregate_result.map.begin(), aggregate_result.map.end(),
            host_map.begin());
        auto coarse_host = copy_device_weighted(coarse_device);
        const auto snapshot = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - snapshot_start).count();
        snapshot_seconds += snapshot;
        std::cout << "ml_gpu_device_snapshot_seconds level=" << level
                  << " seconds=" << snapshot << '\n';

        if (verify) {
            const auto verify_start = std::chrono::steady_clock::now();
            validate_coarsening_step(
                hierarchy.levels.back(), coarse_host, host_map,
                aggregate_result.capacity, level, options.strict_verify);
            std::cout << "ml_gpu_device_verify_seconds level=" << level
                      << " seconds=" << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - verify_start).count()
                      << '\n';
        }
        hierarchy.fine_to_coarse.push_back(std::move(host_map));
        hierarchy.levels.push_back(std::move(coarse_host));
        current = std::move(coarse_device);
    }
    if (level == options.max_levels && current.vertices() > cutoff) {
        hierarchy.stop_reason = "max_levels";
    }
    hierarchy.hierarchy_loop_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - core_start).count();
    hierarchy.snapshot_seconds = snapshot_seconds;
    std::cout << "ml_gpu_device_core_seconds=" << device_algorithm_seconds
              << " aggregate_plus_contract=1 method=sclp\n";
    std::cout << "ml_gpu_timing_total"
              << " affinity_seconds=" << affinity_gpu_seconds
              << " admission_seconds=" << admission_gpu_seconds
              << " two_hop_seconds=" << two_hop_gpu_seconds
              << " compact_seconds=" << compact_gpu_seconds
              << " aggregate_other_seconds=" << std::max(
                     0.0, aggregate_gpu_seconds - affinity_gpu_seconds -
                     admission_gpu_seconds - two_hop_gpu_seconds -
                     compact_gpu_seconds)
              << " aggregate_total_seconds=" << aggregate_gpu_seconds
              << " contraction_seconds=" << contraction_gpu_seconds << '\n';
    std::cout << "ml_gpu_contraction_timing_total"
              << " compact_seconds=" << contraction_compact_gpu_seconds
              << " sort_seconds=" << contraction_sort_gpu_seconds
              << " reduce_seconds=" << contraction_reduce_gpu_seconds
              << " csr_build_seconds=" << contraction_csr_gpu_seconds
              << " vertex_weight_seconds="
              << contraction_vertex_weight_gpu_seconds
              << " total_seconds=" << contraction_gpu_seconds << '\n';
    std::cout << "ml_wall_timing_total"
              << " host_graph_build_seconds=" << host_graph_build_seconds
              << " input_verify_seconds=" << input_verify_seconds
              << " device_input_seconds=" << device_input_seconds
              << " aggregate_seconds=" << aggregate_wall_seconds
              << " contraction_seconds=" << contraction_wall_seconds
              << " snapshot_seconds=" << snapshot_seconds << '\n';
    std::cout << "ml_hierarchy_loop_seconds="
              << hierarchy.hierarchy_loop_seconds
              << " includes_snapshot_and_verify=1 method=sclp\n";
    return hierarchy;
}

template class SclpWorkspace<ActiveTypes>;
template DeviceAggregateResult<ActiveTypes> aggregate<ActiveTypes>(
    const DeviceWeightedGraph<ActiveTypes>&, SclpWorkspace<ActiveTypes>&,
    int, std::uint32_t, int, SclpStats&, bool);
template DeviceWeightedGraph<ActiveTypes> contract<ActiveTypes>(
    DeviceWeightedGraph<ActiveTypes>, DeviceAggregateResult<ActiveTypes>&&,
    SclpWorkspace<ActiveTypes>&, ContractionTimings&);
template Hierarchy<ActiveTypes> coarsen<ActiveTypes>(
    const WeightedGraph<ActiveTypes>&, const CoarsenOptions&);

}  // namespace gpart
