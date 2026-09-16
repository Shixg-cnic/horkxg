#include "refine.hpp"

#include "check.hpp"
#include "graph_types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
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

struct PairAdmissionKey {
    std::uint32_t target;
    std::uint64_t reverse_gain;
    std::uint32_t reverse_tie;
    std::uint32_t owner;
    std::uint32_t partner;
    std::uint32_t source;
    std::uint64_t combined_weight;
};

struct PairAdmissionKeyDecomposer {
    __host__ __device__ auto operator()(PairAdmissionKey& key) const {
        return cuda::std::tie(
            key.target, key.reverse_gain, key.reverse_tie, key.owner);
    }
};

struct ValidPairProposal {
    __host__ __device__ bool operator()(const PairAdmissionKey& key) const {
        return key.target != kInvalidTarget;
    }
};

__device__ __forceinline__ std::uint32_t pair_tie(
    std::uint32_t first, std::uint32_t second,
    std::uint32_t target, std::uint32_t salt) {
    const auto low = min(first, second);
    const auto high = max(first, second);
    return refine_mix32(
        refine_mix32(low) ^ refine_mix32(high) ^
        refine_mix32(target) ^ salt);
}

template <typename OffsetT, typename WeightT, typename VertexT>
__global__ void pair_target_kernel(
    std::int64_t n,
    const OffsetT* offsets,
    const VertexT* neighbors,
    const WeightT* edge_weights,
    const VertexT* partition,
    int parts,
    std::uint32_t tie_salt,
    std::uint32_t* targets,
    std::int64_t* signed_gains) {
    extern __shared__ unsigned long long shared_affinity[];
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) *
        kWarpsPerBlock + warp_in_block;
    if (vertex >= n) return;
    auto* affinity = shared_affinity + warp_in_block * kWarpSize;
    affinity[lane] = 0;
    __syncwarp();

    const auto source = partition[vertex];
    bool boundary = false;
    for (auto edge = offsets[vertex] + lane;
         edge < offsets[vertex + 1]; edge += kWarpSize) {
        const auto neighbor_part = partition[neighbors[edge]];
        boundary = boundary || neighbor_part != source;
        atomicAdd(
            affinity + static_cast<int>(neighbor_part),
            static_cast<unsigned long long>(edge_weights[edge]));
    }
    const unsigned boundary_mask = __ballot_sync(0xffffffffU, boundary);
    __syncwarp();

    if (lane == 0) {
        std::uint32_t target = kInvalidTarget;
        std::int64_t gain = 0;
        if (boundary_mask != 0) {
            unsigned long long best_affinity = 0;
            std::uint32_t best_tie = 0;
            int best_target = -1;
            for (int candidate = 0; candidate < parts; ++candidate) {
                if (candidate == source) continue;
                const auto value = affinity[candidate];
                const auto tie = refine_mix32(
                    static_cast<std::uint32_t>(vertex) ^
                    refine_mix32(static_cast<std::uint32_t>(candidate)) ^
                    tie_salt);
                if (value > best_affinity ||
                    (value == best_affinity && value != 0 &&
                     (tie > best_tie ||
                      (tie == best_tie && candidate < best_target)))) {
                    best_affinity = value;
                    best_tie = tie;
                    best_target = candidate;
                }
            }
            if (best_target >= 0) {
                target = static_cast<std::uint32_t>(best_target);
                const auto current = affinity[source];
                gain = best_affinity >= current
                    ? static_cast<std::int64_t>(best_affinity - current)
                    : -static_cast<std::int64_t>(current - best_affinity);
            }
        }
        targets[vertex] = target;
        signed_gains[vertex] = gain;
    }
}

template <typename OffsetT, typename WeightT, typename VertexT>
__global__ void pair_partner_kernel(
    std::int64_t n,
    const OffsetT* offsets,
    const VertexT* neighbors,
    const WeightT* edge_weights,
    const VertexT* partition,
    const std::uint32_t* targets,
    const std::int64_t* signed_gains,
    std::uint32_t tie_salt,
    VertexT* best_partners,
    std::int64_t* best_pair_gains,
    std::uint64_t* candidate_counts) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (vertex >= n) return;
    const auto source = partition[vertex];
    const auto target = targets[vertex];
    VertexT best_partner = VertexT{-1};
    std::int64_t best_gain = 0;
    std::uint32_t best_tie = 0;
    std::uint64_t candidates = 0;
    if (target != kInvalidTarget) {
        for (auto edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
            const auto neighbor = neighbors[edge];
            if (neighbor == vertex || partition[neighbor] != source ||
                targets[neighbor] != target ||
                (signed_gains[vertex] > 0 && signed_gains[neighbor] > 0)) {
                continue;
            }
            const auto pair_gain = signed_gains[vertex] +
                signed_gains[neighbor] +
                2 * static_cast<std::int64_t>(edge_weights[edge]);
            if (pair_gain <= 0) continue;
            ++candidates;
            const auto tie = pair_tie(
                static_cast<std::uint32_t>(vertex),
                static_cast<std::uint32_t>(neighbor), target, tie_salt);
            if (pair_gain > best_gain ||
                (pair_gain == best_gain &&
                 (tie > best_tie ||
                  (tie == best_tie && neighbor < best_partner)))) {
                best_gain = pair_gain;
                best_tie = tie;
                best_partner = neighbor;
            }
        }
    }
    best_partners[vertex] = best_partner;
    best_pair_gains[vertex] = best_gain;
    candidate_counts[vertex] = candidates;
}

template <typename WeightT, typename VertexT>
__global__ void mutual_pair_proposal_kernel(
    std::int64_t n,
    const VertexT* partition,
    const WeightT* vertex_weights,
    const std::uint32_t* targets,
    const VertexT* best_partners,
    const std::int64_t* best_pair_gains,
    std::uint32_t tie_salt,
    PairAdmissionKey* proposals) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (vertex >= n) return;
    PairAdmissionKey proposal{
        kInvalidTarget, ~std::uint64_t{0}, ~std::uint32_t{0},
        static_cast<std::uint32_t>(vertex), 0, 0, 0};
    const auto partner = best_partners[vertex];
    if (partner >= 0 && vertex < partner &&
        best_partners[partner] == vertex) {
        const auto target = targets[vertex];
        const auto combined =
            static_cast<std::uint64_t>(vertex_weights[vertex]) +
            static_cast<std::uint64_t>(vertex_weights[partner]);
        proposal.target = target;
        proposal.reverse_gain = ~static_cast<std::uint64_t>(
            best_pair_gains[vertex]);
        proposal.reverse_tie = ~pair_tie(
            static_cast<std::uint32_t>(vertex),
            static_cast<std::uint32_t>(partner), target, tie_salt);
        proposal.partner = static_cast<std::uint32_t>(partner);
        proposal.source = static_cast<std::uint32_t>(partition[vertex]);
        proposal.combined_weight = combined;
    }
    proposals[vertex] = proposal;
}

__global__ void decode_pair_proposals_kernel(
    std::int64_t count,
    const PairAdmissionKey* proposals,
    std::int32_t* targets,
    std::uint64_t* weights) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (i >= count) return;
    targets[i] = static_cast<std::int32_t>(proposals[i].target);
    weights[i] = proposals[i].combined_weight;
}

template <typename VertexT>
__global__ void pair_commit_kernel(
    std::int64_t count,
    const PairAdmissionKey* proposals,
    const std::int32_t* accepted,
    VertexT* partition,
    unsigned long long* part_weights) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (i >= count || accepted[i] == 0) return;
    const auto proposal = proposals[i];
    const auto target = static_cast<VertexT>(proposal.target);
    partition[proposal.owner] = target;
    partition[proposal.partner] = target;
    atomicAdd(part_weights + proposal.source, 0ULL - proposal.combined_weight);
    atomicAdd(part_weights + proposal.target, proposal.combined_weight);
}

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

template <typename VertexT>
__global__ void project_partition_kernel(
    std::int64_t n,
    const VertexT* fine_to_coarse,
    const VertexT* coarse_partition,
    VertexT* fine_partition) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (vertex < n) {
        fine_partition[vertex] = coarse_partition[fine_to_coarse[vertex]];
    }
}

template <typename WeightT, typename VertexT>
__global__ void accumulate_part_weights_kernel(
    std::int64_t n,
    const WeightT* vertex_weights,
    const VertexT* partition,
    unsigned long long* part_weights) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) *
        blockDim.x + threadIdx.x;
    if (vertex < n) {
        atomicAdd(
            part_weights + partition[vertex],
            static_cast<unsigned long long>(vertex_weights[vertex]));
    }
}

template <typename Types>
std::uint64_t device_cut_value(
    const DeviceWeightedGraph<Types>& graph,
    const thrust::device_vector<typename Types::VertexT>& partition,
    thrust::device_vector<unsigned long long>& cut_counter) {
    const auto n = graph.vertices();
    const int blocks = static_cast<int>((n + 255) / 256);
    thrust::fill(cut_counter.begin(), cut_counter.end(), 0ULL);
    refine_cut_kernel<<<blocks, 256>>>(
        n,
        thrust::raw_pointer_cast(graph.offsets.data()),
        thrust::raw_pointer_cast(graph.neighbors.data()),
        thrust::raw_pointer_cast(graph.edge_weights.data()),
        thrust::raw_pointer_cast(partition.data()),
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
}

template <typename Types>
void device_part_weights(
    const DeviceWeightedGraph<Types>& graph,
    const thrust::device_vector<typename Types::VertexT>& partition,
    thrust::device_vector<unsigned long long>& weights,
    int parts) {
    weights.resize(static_cast<std::size_t>(parts));
    thrust::fill(weights.begin(), weights.end(), 0ULL);
    const auto n = graph.vertices();
    const int blocks = static_cast<int>((n + 255) / 256);
    accumulate_part_weights_kernel<<<blocks, 256>>>(
        n,
        thrust::raw_pointer_cast(graph.vertex_weights.data()),
        thrust::raw_pointer_cast(partition.data()),
        thrust::raw_pointer_cast(weights.data()));
    CUDA_CHECK(cudaGetLastError());
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
class RefineDeviceContext {
public:
    using VertexT = typename Types::VertexT;
    const WeightedGraph<Types>& host_graph;
    DeviceWeightedGraph<Types> graph;
    thrust::device_vector<VertexT> partition;
    thrust::device_vector<VertexT> previous_partition;
    thrust::device_vector<unsigned long long> part_weights;
    thrust::device_vector<unsigned long long> previous_part_weights;
    thrust::device_vector<unsigned long long> cut_counter;
    thrust::device_vector<RefineAdmissionKey> proposal_keys;
    thrust::device_vector<RefineAdmissionKey> compact_keys;
    thrust::device_vector<RefineAdmissionKey> sorted_keys;
    thrust::device_vector<PairAdmissionKey> pair_proposal_keys;
    thrust::device_vector<PairAdmissionKey> pair_compact_keys;
    thrust::device_vector<PairAdmissionKey> pair_sorted_keys;
    thrust::device_vector<std::uint32_t> pair_targets;
    thrust::device_vector<std::int64_t> signed_gains;
    thrust::device_vector<VertexT> best_partners;
    thrust::device_vector<std::int64_t> best_pair_gains;
    thrust::device_vector<std::uint64_t> candidate_counts;
    thrust::device_vector<std::int32_t> ordered_targets;
    thrust::device_vector<std::uint64_t> ordered_weights;
    thrust::device_vector<std::uint64_t> prefix_weights;
    thrust::device_vector<std::int32_t> accepted_flags;
    thrust::device_vector<std::uint8_t> radix_temp;
    std::vector<std::uint64_t> host_part_weights;
    std::uint64_t capacity;
    std::int64_t n;
    int vertex_blocks;
    int warp_blocks;

    RefineDeviceContext(
        const WeightedGraph<Types>& input,
        const std::vector<VertexT>& labels,
        const RefineOptions& options)
        : host_graph(input),
          n(input.vertices()),
          vertex_blocks(static_cast<int>((n + 255) / 256)),
          warp_blocks(static_cast<int>(
              (n + kWarpsPerBlock - 1) / kWarpsPerBlock)) {
        if (options.parts < 2 || options.parts > 32 ||
            options.max_rounds < 0 || options.max_rounds > 4 ||
            !std::isfinite(options.imbalance_ratio) ||
            options.imbalance_ratio < 1.0 || n <= 0) {
            throw std::invalid_argument("invalid refinement options");
        }
        if (input.offsets.size() != input.vertex_weights.size() + 1 ||
            input.offsets.empty() || input.offsets.front() != 0 ||
            input.offsets.back() != input.edges() ||
            input.neighbors.size() != input.edge_weights.size()) {
            throw std::invalid_argument("invalid weighted CSR for refinement");
        }
        std::uint64_t total_weight = 0;
        host_part_weights = validate_and_measure_partition(
            input, labels, options.parts, total_weight);
        capacity = refinement_capacity(
            total_weight, options.parts, options.imbalance_ratio);
        if (*std::max_element(
                host_part_weights.begin(), host_part_weights.end()) >
            capacity) {
            throw std::invalid_argument(
                "initial refinement partition is imbalanced");
        }
        graph = make_device_weighted(input);
        partition = labels;
        part_weights.assign(
            host_part_weights.begin(), host_part_weights.end());
        initialize_buffers();
    }

    RefineDeviceContext(
        const WeightedGraph<Types>& input,
        DeviceWeightedGraph<Types>&& device_graph,
        thrust::device_vector<VertexT>&& device_partition,
        thrust::device_vector<unsigned long long>&& device_part_weights,
        std::vector<std::uint64_t> measured_part_weights,
        const RefineOptions& options)
        : host_graph(input),
          graph(std::move(device_graph)),
          partition(std::move(device_partition)),
          part_weights(std::move(device_part_weights)),
          host_part_weights(std::move(measured_part_weights)),
          n(input.vertices()),
          vertex_blocks(static_cast<int>((n + 255) / 256)),
          warp_blocks(static_cast<int>(
              (n + kWarpsPerBlock - 1) / kWarpsPerBlock)) {
        std::uint64_t total_weight = 0;
        for (const auto weight : host_part_weights) {
            if (total_weight >
                std::numeric_limits<std::uint64_t>::max() - weight) {
                throw std::overflow_error("refinement vertex weight overflow");
            }
            total_weight += weight;
        }
        capacity = refinement_capacity(
            total_weight, options.parts, options.imbalance_ratio);
        if (partition.size() != static_cast<std::size_t>(n) ||
            part_weights.size() != static_cast<std::size_t>(options.parts) ||
            *std::max_element(
                host_part_weights.begin(), host_part_weights.end()) >
                capacity) {
            throw std::invalid_argument("invalid resident refinement state");
        }
        initialize_buffers();
    }

    void initialize_buffers() {
        previous_partition.resize(static_cast<std::size_t>(n));
        previous_part_weights.resize(part_weights.size());
        cut_counter.resize(1);
        proposal_keys.resize(static_cast<std::size_t>(n));
        compact_keys.resize(static_cast<std::size_t>(n));
    }

    std::uint64_t cut() {
        return device_cut_value(graph, partition, cut_counter);
    }

    void sync_part_weights() {
        thrust::copy(
            part_weights.begin(), part_weights.end(),
            host_part_weights.begin());
    }

    void reserve_radix(std::size_t bytes) {
        if (radix_temp.size() < bytes) radix_temp.resize(bytes);
    }

    void reserve_admission(std::size_t count) {
        if (ordered_targets.size() < count) ordered_targets.resize(count);
        if (ordered_weights.size() < count) ordered_weights.resize(count);
        if (prefix_weights.size() < count) prefix_weights.resize(count);
        if (accepted_flags.size() < count) accepted_flags.resize(count);
    }

    void reserve_plain_sorted(std::size_t count) {
        if (sorted_keys.size() < count) sorted_keys.resize(count);
        reserve_admission(count);
    }

    void reserve_pair_vertices() {
        const auto count = static_cast<std::size_t>(n);
        pair_targets.resize(count);
        signed_gains.resize(count);
        best_partners.resize(count);
        best_pair_gains.resize(count);
        candidate_counts.resize(count);
        pair_proposal_keys.resize(count);
        pair_compact_keys.resize(count);
    }

    void reserve_pair_sorted(std::size_t count) {
        if (pair_sorted_keys.size() < count) pair_sorted_keys.resize(count);
        reserve_admission(count);
    }

    void export_partition(std::vector<VertexT>& labels) {
        thrust::copy(partition.begin(), partition.end(), labels.begin());
    }
};

template <typename Types>
RefineStats run_plain_refinement(
    RefineDeviceContext<Types>& context,
    const RefineOptions& options,
    int rounds) {
    const auto start = std::chrono::steady_clock::now();
    auto& device_graph = context.graph;
    auto& device_partition = context.partition;
    auto& previous_partition = context.previous_partition;
    auto& part_weights = context.part_weights;
    auto& previous_part_weights = context.previous_part_weights;
    auto& proposal_keys = context.proposal_keys;
    auto& compact_keys = context.compact_keys;
    auto& sorted_keys = context.sorted_keys;
    auto& ordered_targets = context.ordered_targets;
    auto& ordered_weights = context.ordered_weights;
    auto& prefix_weights = context.prefix_weights;
    auto& accepted_flags = context.accepted_flags;
    auto& radix_temp = context.radix_temp;
    auto& host_part_weights = context.host_part_weights;
    const auto capacity = context.capacity;
    const auto n = context.n;
    const auto vertex_blocks = context.vertex_blocks;
    const auto warp_blocks = context.warp_blocks;
    const auto device_cut = [&]() { return context.cut(); };

    RefineStats stats;
    stats.initial_cut = device_cut();
    auto current_cut = stats.initial_cut;
    for (int round = 0; round < rounds; ++round) {
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
        context.reserve_plain_sorted(
            static_cast<std::size_t>(proposal_count));

        std::size_t radix_bytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            nullptr, radix_bytes,
            thrust::raw_pointer_cast(compact_keys.data()),
            thrust::raw_pointer_cast(sorted_keys.data()),
            static_cast<int>(proposal_count),
            RefineAdmissionKeyDecomposer{}));
        context.reserve_radix(radix_bytes);
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
        context.sync_part_weights();
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

    context.sync_part_weights();
    stats.final_cut = device_cut();
    stats.final_max_part_weight = *std::max_element(
        host_part_weights.begin(), host_part_weights.end());
    if (stats.final_cut > stats.initial_cut ||
        stats.final_max_part_weight > capacity) {
        throw std::runtime_error("refinement postcondition failed");
    }
    stats.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return stats;
}

template <typename Types>
PairRefineStats run_pair_escape(
    RefineDeviceContext<Types>& context,
    const RefineOptions& options,
    int level) {
    using VertexT = typename Types::VertexT;
    const auto& graph = context.host_graph;
    std::uint64_t directed_weight = 0;
    for (const auto edge_weight : graph.edge_weights) {
        const auto weight = static_cast<std::uint64_t>(edge_weight);
        const auto signed_limit = static_cast<std::uint64_t>(
            std::numeric_limits<std::int64_t>::max()) / 2;
        if (weight > signed_limit ||
            directed_weight > signed_limit - weight) {
            throw std::overflow_error(
                "pair refinement signed gain range exceeded");
        }
        directed_weight += weight;
    }

    auto& host_part_weights = context.host_part_weights;
    const auto capacity = context.capacity;
    if (*std::max_element(host_part_weights.begin(), host_part_weights.end()) >
        capacity) {
        throw std::invalid_argument("initial pair partition is imbalanced");
    }

    auto& device_graph = context.graph;
    auto& device_partition = context.partition;
    auto& previous_partition = context.previous_partition;
    auto& part_weights = context.part_weights;
    auto& previous_part_weights = context.previous_part_weights;
    const auto n = context.n;
    const auto vertex_blocks = context.vertex_blocks;
    const auto warp_blocks = context.warp_blocks;
    const auto device_cut = [&]() { return context.cut(); };
    thrust::copy(
        device_partition.begin(), device_partition.end(),
        previous_partition.begin());
    thrust::copy(
        part_weights.begin(), part_weights.end(),
        previous_part_weights.begin());

    PairRefineStats stats;
    stats.cut_before = device_cut();
    stats.cut_after = stats.cut_before;
    context.reserve_pair_vertices();
    auto& targets = context.pair_targets;
    auto& signed_gains = context.signed_gains;
    const auto tie_salt = options.seed ^
        (static_cast<std::uint32_t>(level) * 0x9e3779b9U) ^ 0x27d4eb2dU;
    constexpr std::size_t shared_bytes =
        kWarpsPerBlock * kWarpSize * sizeof(unsigned long long);
    pair_target_kernel<<<warp_blocks, 256, shared_bytes>>>(
        n,
        thrust::raw_pointer_cast(device_graph.offsets.data()),
        thrust::raw_pointer_cast(device_graph.neighbors.data()),
        thrust::raw_pointer_cast(device_graph.edge_weights.data()),
        thrust::raw_pointer_cast(device_partition.data()),
        options.parts, tie_salt,
        thrust::raw_pointer_cast(targets.data()),
        thrust::raw_pointer_cast(signed_gains.data()));
    CUDA_CHECK(cudaGetLastError());

    auto& best_partners = context.best_partners;
    auto& best_pair_gains = context.best_pair_gains;
    auto& candidate_counts = context.candidate_counts;
    pair_partner_kernel<<<vertex_blocks, 256>>>(
        n,
        thrust::raw_pointer_cast(device_graph.offsets.data()),
        thrust::raw_pointer_cast(device_graph.neighbors.data()),
        thrust::raw_pointer_cast(device_graph.edge_weights.data()),
        thrust::raw_pointer_cast(device_partition.data()),
        thrust::raw_pointer_cast(targets.data()),
        thrust::raw_pointer_cast(signed_gains.data()), tie_salt,
        thrust::raw_pointer_cast(best_partners.data()),
        thrust::raw_pointer_cast(best_pair_gains.data()),
        thrust::raw_pointer_cast(candidate_counts.data()));
    CUDA_CHECK(cudaGetLastError());
    const auto directed_candidates = thrust::reduce(
        candidate_counts.begin(), candidate_counts.end(), std::uint64_t{0});
    if ((directed_candidates & 1ULL) != 0) {
        throw std::runtime_error("pair candidates are not symmetric");
    }
    stats.candidates = directed_candidates / 2;

    auto& proposal_keys = context.pair_proposal_keys;
    auto& compact_keys = context.pair_compact_keys;
    auto& sorted_keys = context.pair_sorted_keys;
    mutual_pair_proposal_kernel<<<vertex_blocks, 256>>>(
        n,
        thrust::raw_pointer_cast(device_partition.data()),
        thrust::raw_pointer_cast(device_graph.vertex_weights.data()),
        thrust::raw_pointer_cast(targets.data()),
        thrust::raw_pointer_cast(best_partners.data()),
        thrust::raw_pointer_cast(best_pair_gains.data()), tie_salt,
        thrust::raw_pointer_cast(proposal_keys.data()));
    CUDA_CHECK(cudaGetLastError());
    const auto compact_end = thrust::copy_if(
        thrust::device, proposal_keys.begin(), proposal_keys.end(),
        compact_keys.begin(), ValidPairProposal{});
    const auto pair_count = static_cast<std::int64_t>(
        compact_end - compact_keys.begin());
    stats.mutual_pairs = static_cast<std::uint64_t>(pair_count);

    if (pair_count > 0) {
        context.reserve_pair_sorted(static_cast<std::size_t>(pair_count));
        std::size_t radix_bytes = 0;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            nullptr, radix_bytes,
            thrust::raw_pointer_cast(compact_keys.data()),
            thrust::raw_pointer_cast(sorted_keys.data()),
            static_cast<int>(pair_count), PairAdmissionKeyDecomposer{}));
        context.reserve_radix(radix_bytes);
        auto call_bytes = radix_bytes;
        CUDA_CHECK(cub::DeviceRadixSort::SortKeys(
            thrust::raw_pointer_cast(context.radix_temp.data()), call_bytes,
            thrust::raw_pointer_cast(compact_keys.data()),
            thrust::raw_pointer_cast(sorted_keys.data()),
            static_cast<int>(pair_count), PairAdmissionKeyDecomposer{}));

        auto& ordered_targets = context.ordered_targets;
        auto& ordered_weights = context.ordered_weights;
        auto& prefix_weights = context.prefix_weights;
        auto& accepted_flags = context.accepted_flags;
        const int pair_blocks = static_cast<int>((pair_count + 255) / 256);
        decode_pair_proposals_kernel<<<pair_blocks, 256>>>(
            pair_count, thrust::raw_pointer_cast(sorted_keys.data()),
            thrust::raw_pointer_cast(ordered_targets.data()),
            thrust::raw_pointer_cast(ordered_weights.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::inclusive_scan_by_key(
            thrust::device,
            ordered_targets.begin(), ordered_targets.begin() + pair_count,
            ordered_weights.begin(), prefix_weights.begin());
        refine_admission_kernel<<<pair_blocks, 256>>>(
            pair_count,
            thrust::raw_pointer_cast(ordered_targets.data()),
            thrust::raw_pointer_cast(prefix_weights.data()),
            thrust::raw_pointer_cast(part_weights.data()), capacity,
            thrust::raw_pointer_cast(accepted_flags.data()));
        CUDA_CHECK(cudaGetLastError());
        stats.accepted = static_cast<std::uint64_t>(thrust::count(
            accepted_flags.begin(), accepted_flags.begin() + pair_count,
            std::int32_t{1}));

        if (stats.accepted > 0) {
            pair_commit_kernel<<<pair_blocks, 256>>>(
                pair_count, thrust::raw_pointer_cast(sorted_keys.data()),
                thrust::raw_pointer_cast(accepted_flags.data()),
                thrust::raw_pointer_cast(device_partition.data()),
                thrust::raw_pointer_cast(part_weights.data()));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            context.sync_part_weights();
            if (*std::max_element(
                    host_part_weights.begin(), host_part_weights.end()) >
                capacity) {
                throw std::runtime_error(
                    "pair admission violated partition capacity");
            }
            stats.cut_after = device_cut();
            if (stats.cut_after > stats.cut_before) {
                stats.rollback = true;
                thrust::copy(
                    previous_partition.begin(), previous_partition.end(),
                    device_partition.begin());
                thrust::copy(
                    previous_part_weights.begin(), previous_part_weights.end(),
                    part_weights.begin());
            }
        }
    }

    return stats;
}

template <typename Types>
void refine_partition(
    const WeightedGraph<Types>& graph,
    std::vector<typename Types::VertexT>& partition,
    const RefineOptions& options,
    RefineStats* output_stats) {
    RefineDeviceContext<Types> context(graph, partition, options);
    auto stats = run_plain_refinement(context, options, options.max_rounds);
    context.export_partition(partition);
    if (output_stats != nullptr) *output_stats = stats;
}

template <typename Types>
void coordinated_pair_escape(
    const WeightedGraph<Types>& graph,
    std::vector<typename Types::VertexT>& partition,
    const RefineOptions& options,
    int level,
    PairRefineStats* output_stats) {
    RefineDeviceContext<Types> context(graph, partition, options);
    auto stats = run_pair_escape(context, options, level);
    context.export_partition(partition);
    if (output_stats != nullptr) *output_stats = stats;
}

template <typename Types>
DeviceUncoarsenResult<Types> refine_hierarchy_device(
    const Hierarchy<Types>& hierarchy,
    const std::vector<typename Types::VertexT>& coarsest_partition,
    const RefineOptions& options) {
    using VertexT = typename Types::VertexT;
    if (hierarchy.levels.empty() ||
        hierarchy.fine_to_coarse.size() + 1 != hierarchy.levels.size()) {
        throw std::invalid_argument("resident refinement requires a complete hierarchy");
    }
    const auto& coarsest = hierarchy.levels.back();
    std::uint64_t total_weight = 0;
    auto current_host_weights = validate_and_measure_partition(
        coarsest, coarsest_partition, options.parts, total_weight);

    const auto initial_h2d_start = std::chrono::steady_clock::now();
    auto current_graph = make_device_weighted(coarsest);
    thrust::device_vector<VertexT> current_partition = coarsest_partition;
    thrust::device_vector<unsigned long long> current_part_weights(
        current_host_weights.begin(), current_host_weights.end());
    double pending_graph_h2d = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initial_h2d_start).count();
    thrust::device_vector<unsigned long long> cut_counter(1);

    DeviceUncoarsenResult<Types> result;
    result.levels.reserve(hierarchy.levels.size() - 1);
    std::uint64_t last_cut = device_cut_value(
        current_graph, current_partition, cut_counter);

    for (std::size_t coarse_level = hierarchy.levels.size() - 1;
         coarse_level > 0; --coarse_level) {
        const auto fine_level = coarse_level - 1;
        const auto& fine = hierarchy.levels[fine_level];
        const auto& map = hierarchy.fine_to_coarse[fine_level];
        if (map.size() != static_cast<std::size_t>(fine.vertices())) {
            throw std::invalid_argument("fine-to-coarse map has wrong length");
        }
        for (const auto coarse_vertex : map) {
            if (coarse_vertex < 0 ||
                coarse_vertex >= hierarchy.levels[coarse_level].vertices()) {
                throw std::invalid_argument(
                    "fine-to-coarse map contains an invalid id");
            }
        }

        RefineLevelResult report;
        report.level = fine_level;
        report.vertices = fine.vertices();
        report.edge_entries = fine.edges();
        const auto graph_h2d_start = std::chrono::steady_clock::now();
        auto fine_graph = make_device_weighted(fine);
        thrust::device_vector<VertexT> device_map = map;
        thrust::device_vector<VertexT> fine_partition(
            static_cast<std::size_t>(fine.vertices()));
        report.timings.graph_h2d_seconds = pending_graph_h2d +
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - graph_h2d_start).count();
        pending_graph_h2d = 0.0;

        const auto projection_start = std::chrono::steady_clock::now();
        const int projection_blocks = static_cast<int>(
            (fine.vertices() + 255) / 256);
        project_partition_kernel<<<projection_blocks, 256>>>(
            fine.vertices(), thrust::raw_pointer_cast(device_map.data()),
            thrust::raw_pointer_cast(current_partition.data()),
            thrust::raw_pointer_cast(fine_partition.data()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        report.timings.projection_gpu_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - projection_start).count();

        const auto verification_start = std::chrono::steady_clock::now();
        thrust::device_vector<unsigned long long> fine_part_weights;
        device_part_weights(
            fine_graph, fine_partition, fine_part_weights, options.parts);
        std::vector<std::uint64_t> projected_weights(
            static_cast<std::size_t>(options.parts));
        thrust::copy(
            fine_part_weights.begin(), fine_part_weights.end(),
            projected_weights.begin());
        report.projection_cut = device_cut_value(
            fine_graph, fine_partition, cut_counter);
        if (options.strict_verify) {
            const auto coarse_cut = device_cut_value(
                current_graph, current_partition, cut_counter);
            std::vector<std::uint64_t> coarse_weights(
                static_cast<std::size_t>(options.parts));
            thrust::copy(
                current_part_weights.begin(), current_part_weights.end(),
                coarse_weights.begin());
            if (coarse_cut != report.projection_cut ||
                coarse_weights != projected_weights) {
                throw std::runtime_error(
                    "GPU projection failed cut or part-weight preservation");
            }
        }
        report.projection_max_part_weight = *std::max_element(
            projected_weights.begin(), projected_weights.end());
        const auto projected_total = std::accumulate(
            projected_weights.begin(), projected_weights.end(),
            std::uint64_t{0});
        report.projection_imbalance = projected_total == 0 ? 0.0 :
            static_cast<double>(report.projection_max_part_weight) *
            options.parts / static_cast<double>(projected_total);
        report.timings.verification_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - verification_start).count();

        const auto context_start = std::chrono::steady_clock::now();
        RefineDeviceContext<Types> context(
            fine, std::move(fine_graph), std::move(fine_partition),
            std::move(fine_part_weights), projected_weights, options);
        report.timings.graph_h2d_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - context_start).count();
        report.plain = run_plain_refinement(
            context, options, options.max_rounds);
        report.timings.plain_seconds = report.plain.seconds;
        const auto pair_start = std::chrono::steady_clock::now();
        report.pair = run_pair_escape(
            context, options, static_cast<int>(fine_level));
        report.timings.pair_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pair_start).count();
        report.cleanup = run_plain_refinement(context, options, 1);
        report.timings.cleanup_seconds = report.cleanup.seconds;
        last_cut = report.cleanup.final_cut;
        current_host_weights = context.host_part_weights;
        current_graph = std::move(context.graph);
        current_partition = std::move(context.partition);
        current_part_weights = std::move(context.part_weights);
        result.levels.push_back(std::move(report));
    }

    result.final_cut = last_cut;
    result.final_part_weights = current_host_weights;
    const auto directed_edge_weight = thrust::reduce(
        current_graph.edge_weights.begin(), current_graph.edge_weights.end(),
        std::uint64_t{0});
    if ((directed_edge_weight & 1ULL) != 0) {
        throw std::runtime_error("symmetric graph has odd directed edge weight");
    }
    result.total_edge_weight = directed_edge_weight / 2;
    result.partition.resize(static_cast<std::size_t>(current_graph.vertices()));
    const auto final_d2h_start = std::chrono::steady_clock::now();
    thrust::copy(
        current_partition.begin(), current_partition.end(),
        result.partition.begin());
    result.final_d2h_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - final_d2h_start).count();
    return result;
}

template void refine_partition<ActiveTypes>(
    const WeightedGraph<ActiveTypes>&,
    std::vector<ActiveTypes::VertexT>&,
    const RefineOptions&,
    RefineStats*);

template void coordinated_pair_escape<ActiveTypes>(
    const WeightedGraph<ActiveTypes>&,
    std::vector<ActiveTypes::VertexT>&,
    const RefineOptions&,
    int,
    PairRefineStats*);

template DeviceUncoarsenResult<ActiveTypes>
refine_hierarchy_device<ActiveTypes>(
    const Hierarchy<ActiveTypes>&,
    const std::vector<ActiveTypes::VertexT>&,
    const RefineOptions&);

}  // namespace gpart
