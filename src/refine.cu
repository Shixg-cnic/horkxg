#include "refine.hpp"

#include "check.hpp"
#include "graph_types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/scan.h>

namespace gpart {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr std::uint32_t kInvalidTarget = 0xffffffffU;

__device__ __forceinline__ std::uint32_t refine_mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

struct RefineAdmissionKey {
    std::uint32_t target;
    std::uint64_t reverse_gain;
    std::uint32_t reverse_tie;
    std::uint32_t vertex;
};

struct RefineAdmissionKeyDecomposer {
    __host__ __device__ auto operator()(RefineAdmissionKey& key) const {
        return cuda::std::tie(
            key.target, key.reverse_gain, key.reverse_tie, key.vertex);
    }
};

struct ValidRefineProposal {
    __host__ __device__ bool operator()(const RefineAdmissionKey& key) const {
        return key.target != kInvalidTarget;
    }
};

template <typename OffsetT, typename WeightT, typename VertexT>
__global__ void refine_proposal_kernel(
    std::int64_t n,
    const OffsetT* offsets,
    const VertexT* neighbors,
    const WeightT* edge_weights,
    const VertexT* partition,
    int parts,
    std::uint32_t tie_salt,
    RefineAdmissionKey* proposals) {
    extern __shared__ unsigned long long shared_affinity[];
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) *
        kWarpsPerBlock + warp_in_block;
    if (vertex >= n) return;
    auto* affinity = shared_affinity + warp_in_block * kWarpSize;
    affinity[lane] = 0;
    __syncwarp();

    const auto current_part = partition[vertex];
    bool boundary = false;
    for (auto edge = offsets[vertex] + lane;
         edge < offsets[vertex + 1]; edge += kWarpSize) {
        const auto neighbor_part = partition[neighbors[edge]];
        boundary = boundary || neighbor_part != current_part;
        atomicAdd(
            affinity + static_cast<int>(neighbor_part),
            static_cast<unsigned long long>(edge_weights[edge]));
    }
    const unsigned boundary_mask = __ballot_sync(0xffffffffU, boundary);
    __syncwarp();

    if (lane == 0) {
        RefineAdmissionKey key{
            kInvalidTarget, ~std::uint64_t{0}, ~std::uint32_t{0},
            static_cast<std::uint32_t>(vertex)};
        if (boundary_mask != 0) {
            const auto current_affinity = affinity[current_part];
            unsigned long long best_affinity = 0;
            std::uint32_t best_tie = 0;
            int best_target = -1;
            for (int target = 0; target < parts; ++target) {
                if (target == current_part) continue;
                const auto value = affinity[target];
                const auto tie = refine_mix32(
                    static_cast<std::uint32_t>(vertex) ^
                    refine_mix32(static_cast<std::uint32_t>(target)) ^
                    tie_salt);
                if (value > best_affinity ||
                    (value == best_affinity && value != 0 &&
                     (tie > best_tie ||
                      (tie == best_tie && target < best_target)))) {
                    best_affinity = value;
                    best_tie = tie;
                    best_target = target;
                }
            }
            if (best_target >= 0 && best_affinity > current_affinity) {
                const auto gain = best_affinity - current_affinity;
                key.target = static_cast<std::uint32_t>(best_target);
                key.reverse_gain = ~static_cast<std::uint64_t>(gain);
                key.reverse_tie = ~best_tie;
            }
        }
        proposals[vertex] = key;
    }
}

template <typename WeightT>
__global__ void decode_refine_proposals_kernel(
    std::int64_t count,
    const RefineAdmissionKey* proposals,
    const WeightT* vertex_weights,
    std::int32_t* targets,
    std::uint64_t* weights) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (i >= count) return;
    targets[i] = static_cast<std::int32_t>(proposals[i].target);
    weights[i] = static_cast<std::uint64_t>(
        vertex_weights[proposals[i].vertex]);
}

__global__ void refine_admission_kernel(
    std::int64_t count,
    const std::int32_t* targets,
    const std::uint64_t* prefix_weights,
    const unsigned long long* part_weights,
    std::uint64_t capacity,
    std::int32_t* accepted) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto target = targets[i];
    const auto base = static_cast<std::uint64_t>(part_weights[target]);
    accepted[i] = base <= capacity && prefix_weights[i] <= capacity - base
        ? 1 : 0;
}

template <typename VertexT, typename WeightT>
__global__ void refine_commit_kernel(
    std::int64_t count,
    const RefineAdmissionKey* proposals,
    const std::int32_t* accepted,
    const WeightT* vertex_weights,
    VertexT* partition,
    unsigned long long* part_weights) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (i >= count || accepted[i] == 0) return;
    const auto vertex = proposals[i].vertex;
    const auto source = partition[vertex];
    const auto target = static_cast<VertexT>(proposals[i].target);
    const auto weight = static_cast<unsigned long long>(vertex_weights[vertex]);
    atomicAdd(part_weights + source, 0ULL - weight);
    atomicAdd(part_weights + target, weight);
    partition[vertex] = target;
}

template <typename OffsetT, typename WeightT, typename VertexT>
__global__ void refine_cut_kernel(
    std::int64_t n,
    const OffsetT* offsets,
    const VertexT* neighbors,
    const WeightT* edge_weights,
    const VertexT* partition,
    unsigned long long* directed_cut) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (vertex >= n) return;
    unsigned long long local = 0;
    for (auto edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
        if (partition[vertex] != partition[neighbors[edge]]) {
            local += static_cast<unsigned long long>(edge_weights[edge]);
        }
    }
    if (local != 0) atomicAdd(directed_cut, local);
}

std::uint64_t refinement_capacity(
    std::uint64_t total_weight, int parts, double imbalance_ratio) {
    const long double raw =
        static_cast<long double>(total_weight) *
        static_cast<long double>(imbalance_ratio) /
        static_cast<long double>(parts);
    if (!std::isfinite(raw) ||
        raw > static_cast<long double>(
                  std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error("refinement capacity exceeds uint64");
    }
    return static_cast<std::uint64_t>(std::ceil(raw));
}

template <typename Types>
std::vector<std::uint64_t> validate_and_measure_partition(
    const WeightedGraph<Types>& graph,
    const std::vector<typename Types::VertexT>& partition,
    int parts,
    std::uint64_t& total_weight) {
    if (partition.size() != static_cast<std::size_t>(graph.vertices())) {
        throw std::invalid_argument("refinement partition has the wrong length");
    }
    std::vector<std::uint64_t> weights(static_cast<std::size_t>(parts), 0);
    total_weight = 0;
    for (std::size_t v = 0; v < partition.size(); ++v) {
        const auto part = partition[v];
        if (part < 0 || part >= parts) {
            throw std::invalid_argument("refinement partition has an invalid part id");
        }
        const auto weight = static_cast<std::uint64_t>(graph.vertex_weights[v]);
        if (weights[part] > std::numeric_limits<std::uint64_t>::max() - weight ||
            total_weight > std::numeric_limits<std::uint64_t>::max() - weight) {
            throw std::overflow_error("refinement vertex weight sum overflow");
        }
        weights[part] += weight;
        total_weight += weight;
    }
    return weights;
}

}  // namespace

template <typename Types>
void refine_partition(
    const WeightedGraph<Types>& graph,
    std::vector<typename Types::VertexT>& partition,
    const RefineOptions& options,
    RefineStats* output_stats) {
    using VertexT = typename Types::VertexT;
    const auto start = std::chrono::steady_clock::now();
    if (options.parts < 2 || options.parts > 32 ||
        options.max_rounds < 0 || options.max_rounds > 4 ||
        !std::isfinite(options.imbalance_ratio) ||
        options.imbalance_ratio < 1.0) {
        throw std::invalid_argument("invalid refinement options");
    }
    if (graph.vertices() <= 0) {
        throw std::invalid_argument("refinement requires a nonempty graph");
    }
    if (graph.offsets.size() != graph.vertex_weights.size() + 1 ||
        graph.offsets.empty() || graph.offsets.front() != 0 ||
        graph.offsets.back() != graph.edges() ||
        graph.neighbors.size() != graph.edge_weights.size()) {
        throw std::invalid_argument("invalid weighted CSR for refinement");
    }

    std::uint64_t total_weight = 0;
    auto host_part_weights = validate_and_measure_partition(
        graph, partition, options.parts, total_weight);
    const auto capacity = refinement_capacity(
        total_weight, options.parts, options.imbalance_ratio);
    if (*std::max_element(host_part_weights.begin(), host_part_weights.end()) >
        capacity) {
        throw std::invalid_argument("initial refinement partition is imbalanced");
    }

    auto device_graph = make_device_weighted(graph);
    thrust::device_vector<VertexT> device_partition = partition;
    thrust::device_vector<VertexT> previous_partition(device_partition.size());
    thrust::device_vector<unsigned long long> part_weights(
        host_part_weights.begin(), host_part_weights.end());
    thrust::device_vector<unsigned long long> previous_part_weights(
        part_weights.size());
    thrust::device_vector<unsigned long long> cut_counter(1);
    thrust::device_vector<RefineAdmissionKey> proposal_keys(
        static_cast<std::size_t>(graph.vertices()));
    thrust::device_vector<RefineAdmissionKey> compact_keys(
        static_cast<std::size_t>(graph.vertices()));
    thrust::device_vector<RefineAdmissionKey> sorted_keys(
        static_cast<std::size_t>(graph.vertices()));
    thrust::device_vector<std::int32_t> ordered_targets(
        static_cast<std::size_t>(graph.vertices()));
    thrust::device_vector<std::uint64_t> ordered_weights(
        static_cast<std::size_t>(graph.vertices()));
    thrust::device_vector<std::uint64_t> prefix_weights(
        static_cast<std::size_t>(graph.vertices()));
    thrust::device_vector<std::int32_t> accepted_flags(
        static_cast<std::size_t>(graph.vertices()));
    thrust::device_vector<std::uint8_t> radix_temp;

    const auto n = graph.vertices();
    const int vertex_blocks = static_cast<int>((n + 255) / 256);
    const int warp_blocks = static_cast<int>(
        (n + kWarpsPerBlock - 1) / kWarpsPerBlock);
    const auto device_cut = [&]() {
        thrust::fill(cut_counter.begin(), cut_counter.end(), 0ULL);
        refine_cut_kernel<<<vertex_blocks, 256>>>(
            n,
            thrust::raw_pointer_cast(device_graph.offsets.data()),
            thrust::raw_pointer_cast(device_graph.neighbors.data()),
            thrust::raw_pointer_cast(device_graph.edge_weights.data()),
            thrust::raw_pointer_cast(device_partition.data()),
            thrust::raw_pointer_cast(cut_counter.data()));
        CUDA_CHECK(cudaGetLastError());
        unsigned long long directed = 0;
        CUDA_CHECK(cudaMemcpy(
            &directed, thrust::raw_pointer_cast(cut_counter.data()),
            sizeof(directed), cudaMemcpyDeviceToHost));
        if ((directed & 1ULL) != 0) {
            throw std::runtime_error("refinement cut is not symmetric");
        }
        return static_cast<std::uint64_t>(directed / 2);
    };

    RefineStats stats;
    stats.initial_cut = device_cut();
    auto current_cut = stats.initial_cut;
    for (int round = 0; round < options.max_rounds; ++round) {
        const auto round_start_cut = current_cut;
        stats.rounds = round + 1;
        thrust::copy(
            device_partition.begin(), device_partition.end(),
            previous_partition.begin());
        thrust::copy(
            part_weights.begin(), part_weights.end(),
            previous_part_weights.begin());
        const auto tie_salt = options.seed ^
            (static_cast<std::uint32_t>(round) * 0x9e3779b9U) ^ 0x85ebca6bU;
        constexpr std::size_t shared_bytes =
            kWarpsPerBlock * kWarpSize * sizeof(unsigned long long);
        refine_proposal_kernel<<<warp_blocks, 256, shared_bytes>>>(
            n,
            thrust::raw_pointer_cast(device_graph.offsets.data()),
            thrust::raw_pointer_cast(device_graph.neighbors.data()),
            thrust::raw_pointer_cast(device_graph.edge_weights.data()),
            thrust::raw_pointer_cast(device_partition.data()),
            options.parts, tie_salt,
            thrust::raw_pointer_cast(proposal_keys.data()));
        CUDA_CHECK(cudaGetLastError());
        const auto compact_end = thrust::copy_if(
            thrust::device, proposal_keys.begin(), proposal_keys.end(),
            compact_keys.begin(), ValidRefineProposal{});
        const auto proposal_count = static_cast<std::int64_t>(
            compact_end - compact_keys.begin());
        stats.proposals += static_cast<std::uint64_t>(proposal_count);
        if (proposal_count == 0) break;

        std::size_t radix_bytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            nullptr, radix_bytes,
            thrust::raw_pointer_cast(compact_keys.data()),
            thrust::raw_pointer_cast(sorted_keys.data()),
            static_cast<int>(proposal_count),
            RefineAdmissionKeyDecomposer{}));
        radix_temp.resize(radix_bytes);
        auto call_bytes = radix_bytes;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            thrust::raw_pointer_cast(radix_temp.data()), call_bytes,
            thrust::raw_pointer_cast(compact_keys.data()),
            thrust::raw_pointer_cast(sorted_keys.data()),
            static_cast<int>(proposal_count),
            RefineAdmissionKeyDecomposer{}));
        const int proposal_blocks = static_cast<int>(
            (proposal_count + 255) / 256);
        decode_refine_proposals_kernel<<<proposal_blocks, 256>>>(
            proposal_count,
            thrust::raw_pointer_cast(sorted_keys.data()),
            thrust::raw_pointer_cast(device_graph.vertex_weights.data()),
            thrust::raw_pointer_cast(ordered_targets.data()),
            thrust::raw_pointer_cast(ordered_weights.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::inclusive_scan_by_key(
            thrust::device,
            ordered_targets.begin(), ordered_targets.begin() + proposal_count,
            ordered_weights.begin(), prefix_weights.begin());
        refine_admission_kernel<<<proposal_blocks, 256>>>(
            proposal_count,
            thrust::raw_pointer_cast(ordered_targets.data()),
            thrust::raw_pointer_cast(prefix_weights.data()),
            thrust::raw_pointer_cast(part_weights.data()),
            capacity,
            thrust::raw_pointer_cast(accepted_flags.data()));
        CUDA_CHECK(cudaGetLastError());
        const auto accepted_count = static_cast<std::uint64_t>(thrust::count(
            accepted_flags.begin(), accepted_flags.begin() + proposal_count,
            std::int32_t{1}));
        if (accepted_count == 0) break;

        refine_commit_kernel<<<proposal_blocks, 256>>>(
            proposal_count,
            thrust::raw_pointer_cast(sorted_keys.data()),
            thrust::raw_pointer_cast(accepted_flags.data()),
            thrust::raw_pointer_cast(device_graph.vertex_weights.data()),
            thrust::raw_pointer_cast(device_partition.data()),
            thrust::raw_pointer_cast(part_weights.data()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        thrust::copy(
            part_weights.begin(), part_weights.end(), host_part_weights.begin());
        if (*std::max_element(
                host_part_weights.begin(), host_part_weights.end()) > capacity) {
            throw std::runtime_error("refinement admission violated balance");
        }
        const auto next_cut = device_cut();
        if (next_cut > current_cut) {
            thrust::copy(
                previous_partition.begin(), previous_partition.end(),
                device_partition.begin());
            thrust::copy(
                previous_part_weights.begin(), previous_part_weights.end(),
                part_weights.begin());
            break;
        }
        stats.accepted += accepted_count;
        current_cut = next_cut;
        if (next_cut == round_start_cut) break;
    }

    thrust::copy(
        device_partition.begin(), device_partition.end(), partition.begin());
    thrust::copy(
        part_weights.begin(), part_weights.end(), host_part_weights.begin());
    stats.final_cut = device_cut();
    stats.final_max_part_weight = *std::max_element(
        host_part_weights.begin(), host_part_weights.end());
    if (stats.final_cut > stats.initial_cut ||
        stats.final_max_part_weight > capacity) {
        throw std::runtime_error("refinement postcondition failed");
    }
    stats.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (output_stats != nullptr) *output_stats = stats;
}

template void refine_partition<ActiveTypes>(
    const WeightedGraph<ActiveTypes>&,
    std::vector<ActiveTypes::VertexT>&,
    const RefineOptions&,
    RefineStats*);

}  // namespace gpart
