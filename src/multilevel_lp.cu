#include "check.hpp"
#include "graph.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// Quality-first research prototype.  Aggregation and coarse-CSR construction
// deliberately remain on the host until the quality hypothesis is validated;
// weighted refinement is executed on the GPU.
struct WeightedGraph {
    std::vector<std::int64_t> offsets;
    std::vector<std::int32_t> neighbors;
    std::vector<std::uint64_t> edge_weights;
    std::vector<std::uint64_t> vertex_weights;

    std::int64_t vertices() const {
        return static_cast<std::int64_t>(vertex_weights.size());
    }
    std::int64_t edges() const {
        return static_cast<std::int64_t>(neighbors.size());
    }
};

struct EdgeRecord {
    std::uint64_t key;
    std::uint64_t weight;
    bool operator<(const EdgeRecord& other) const { return key < other.key; }
};

struct AggregateResult {
    std::vector<std::int32_t> map;
    std::int32_t coarse_vertices = 0;
    std::uint64_t maximum_weight = 0;
    std::uint64_t capacity = 0;
};

__device__ __forceinline__ std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

__global__ void weighted_propose_kernel(
    std::int64_t n, int parts, int phase, int phases, int round,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::uint64_t* edge_weights, const std::int32_t* labels,
    std::int32_t* targets, std::int64_t* gains) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const int current = labels[v];
    targets[v] = current;
    gains[v] = 0;
    if (static_cast<int>(mix32(static_cast<std::uint32_t>(v) ^
                               static_cast<std::uint32_t>(round * 0x9e3779b9U)) %
                         static_cast<std::uint32_t>(phases)) != phase) {
        return;
    }
    std::int64_t connection[32] = {};
    for (std::int64_t e = offsets[v]; e < offsets[v + 1]; ++e) {
        const int p = labels[neighbors[e]];
        if (p >= 0 && p < parts) {
            connection[p] += static_cast<std::int64_t>(edge_weights[e]);
        }
    }
    int best = current;
    std::int64_t best_gain = 0;
    for (int p = 0; p < parts; ++p) {
        if (p == current || connection[p] == 0) continue;
        const auto gain = connection[p] - connection[current];
        if (gain > best_gain ||
            (gain == best_gain && gain > 0 &&
             mix32(static_cast<std::uint32_t>(v) ^ static_cast<std::uint32_t>(p)) >
             mix32(static_cast<std::uint32_t>(v) ^ static_cast<std::uint32_t>(best)))) {
            best = p;
            best_gain = gain;
        }
    }
    targets[v] = best;
    gains[v] = best_gain;
}

__global__ void weighted_apply_kernel(
    std::int64_t n, const std::uint64_t* vertex_weights,
    const std::int32_t* targets, const std::int64_t* gains,
    unsigned long long capacity, unsigned long long* loads,
    std::int32_t* labels, unsigned long long* changed) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n || gains[v] <= 0) return;
    const int source = labels[v];
    const int target = targets[v];
    if (source == target) return;
    const auto weight = static_cast<unsigned long long>(vertex_weights[v]);
    const auto before = atomicAdd(&loads[target], weight);
    if (before + weight <= capacity) {
        labels[v] = target;
        atomicAdd(&loads[source], static_cast<unsigned long long>(0) - weight);
        atomicAdd(changed, 1ULL);
    } else {
        atomicAdd(&loads[target], static_cast<unsigned long long>(0) - weight);
    }
}

__global__ void weighted_cut_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint64_t* edge_weights,
    const std::int32_t* labels, unsigned long long* cut) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    unsigned long long local = 0;
    const int p = labels[v];
    for (std::int64_t e = offsets[v]; e < offsets[v + 1]; ++e) {
        if (labels[neighbors[e]] != p) local += edge_weights[e];
    }
    if (local) atomicAdd(cut, local);
}

__global__ void build_lp_edge_keys_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint64_t* edge_weights,
    const std::int32_t* clusters, std::uint64_t* keys,
    std::uint64_t* values) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    for (auto e = offsets[v]; e < offsets[v + 1]; ++e) {
        keys[e] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)) << 32) |
                  static_cast<std::uint32_t>(clusters[neighbors[e]]);
        values[e] = edge_weights[e];
    }
}

__global__ void lp_best_connection_kernel(
    std::int64_t count, std::uint64_t cluster_cap,
    const std::uint64_t* keys, const std::uint64_t* connections,
    const std::uint64_t* vertex_weights,
    const std::uint64_t* cluster_weights,
    const std::int32_t* clusters, std::uint64_t* best_connections) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto v = static_cast<std::uint32_t>(keys[i] >> 32);
    const auto target = static_cast<std::uint32_t>(keys[i]);
    const auto source = static_cast<std::uint32_t>(clusters[v]);
    if (target != source && cluster_weights[target] + vertex_weights[v] > cluster_cap) {
        return;
    }
    atomicMax(reinterpret_cast<unsigned long long*>(&best_connections[v]),
              static_cast<unsigned long long>(connections[i]));
}

__global__ void lp_best_target_kernel(
    std::int64_t count, std::uint64_t cluster_cap, int round,
    std::uint32_t seed,
    const std::uint64_t* keys, const std::uint64_t* connections,
    const std::uint64_t* vertex_weights,
    const std::uint64_t* cluster_weights,
    const std::int32_t* clusters, const std::uint64_t* best_connections,
    unsigned long long* best_ties) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto v = static_cast<std::uint32_t>(keys[i] >> 32);
    const auto target = static_cast<std::uint32_t>(keys[i]);
    const auto source = static_cast<std::uint32_t>(clusters[v]);
    if (connections[i] != best_connections[v]) return;
    if (target != source && cluster_weights[target] + vertex_weights[v] > cluster_cap) {
        return;
    }
    const auto tie = static_cast<unsigned long long>(mix32(
        v ^ target ^ seed ^ static_cast<std::uint32_t>(round * 0x9e3779b9U)));
    const auto key = (tie << 32) | (0xffffffffULL - target);
    atomicMax(&best_ties[v], key);
}

__global__ void lp_decode_targets_kernel(
    std::int64_t n, const std::uint64_t* best_connections,
    const unsigned long long* best_ties, const std::int32_t* clusters,
    std::int32_t* targets) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (best_connections[v] == 0) {
        targets[v] = clusters[v];
    } else {
        targets[v] = static_cast<std::int32_t>(
            0xffffffffULL - (best_ties[v] & 0xffffffffULL));
    }
}

__global__ void lp_apply_cluster_moves_kernel(
    std::int64_t n, std::uint64_t cluster_cap,
    const std::uint64_t* vertex_weights, std::uint64_t* cluster_weights,
    const std::int32_t* targets, std::int32_t* clusters,
    unsigned long long* changed) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const auto source = clusters[v];
    const auto target = targets[v];
    if (target < 0 || target == source) return;
    const auto weight = static_cast<unsigned long long>(vertex_weights[v]);
    const auto before = atomicAdd(
        reinterpret_cast<unsigned long long*>(&cluster_weights[target]), weight);
    if (before + weight <= cluster_cap) {
        atomicAdd(reinterpret_cast<unsigned long long*>(&cluster_weights[source]),
                  static_cast<unsigned long long>(0) - weight);
        clusters[v] = target;
        atomicAdd(changed, 1ULL);
    } else {
        atomicAdd(reinterpret_cast<unsigned long long*>(&cluster_weights[target]),
                  static_cast<unsigned long long>(0) - weight);
    }
}

constexpr int BASC_MAX_K = 4;
constexpr std::int32_t BASC_INVALID = -1;
constexpr int BASC_WARPS_PER_BLOCK = 8;
constexpr int BASC_WARP_SIZE = 32;
constexpr std::uint64_t BASC_INVALID_KEY = std::numeric_limits<std::uint64_t>::max();

struct DeviceWeightedGraph {
    thrust::device_vector<std::int64_t> offsets;
    thrust::device_vector<std::int32_t> neighbors;
    thrust::device_vector<std::uint64_t> edge_weights;
    thrust::device_vector<std::uint64_t> vertex_weights;

    std::int64_t vertices() const {
        return static_cast<std::int64_t>(vertex_weights.size());
    }
    std::int64_t edges() const {
        return static_cast<std::int64_t>(neighbors.size());
    }
};

struct DeviceAggregateResult {
    thrust::device_vector<std::int32_t> map;
    thrust::device_vector<std::uint64_t> vertex_weights;
    std::int32_t coarse_vertices = 0;
    std::uint64_t maximum_weight = 0;
    std::uint64_t capacity = 0;
};

struct BascStats {
    std::uint64_t anchors = 0;
    std::uint64_t candidate_histogram[4] = {};
    std::uint64_t first_round_eligible = 0;
    std::uint64_t accepted_first_round = 0;
    std::uint64_t capacity_reject_first_round = 0;
    std::uint64_t expansion_eligible = 0;
    std::uint64_t accepted_expansion = 0;
    std::uint64_t capacity_reject_expansion = 0;
    std::uint64_t singleton_count = 0;
    std::uint64_t cluster_cap = 0;
    double confidence_p50 = 0.0;
    double confidence_p90 = 0.0;
    double anchor_seconds = 0.0;
    double candidate_seconds = 0.0;
    double support_seconds = 0.0;
    double admission_seconds = 0.0;
    double expansion_seconds = 0.0;
};

struct FrontierStats {
    std::int64_t target_clusters = 0;
    std::uint64_t hot_seeds = 0;
    std::uint64_t cold_seeds = 0;
    std::uint64_t cold_seed_rounds = 0;
    std::uint64_t emergency_seeds = 0;
    std::uint64_t proposed = 0;
    std::uint64_t postponed = 0;
    std::uint64_t accepted = 0;
    std::uint64_t capacity_rejected = 0;
    std::uint64_t growth_rounds = 0;
    std::uint64_t boundary_rounds = 0;
    std::uint64_t boundary_moved = 0;
    std::uint64_t boundary_gain = 0;
    std::uint64_t cluster_cap = 0;
    double seed_seconds = 0.0;
    double growth_seconds = 0.0;
    double boundary_seconds = 0.0;
};

struct SclpStats {
    std::uint64_t capacity = 0;
    std::uint64_t lp_accepted = 0;
    std::uint64_t capacity_rejected = 0;
    std::uint64_t two_hop_merged = 0;
    std::uint64_t singleton_count = 0;
    int rounds = 0;
    double affinity_seconds = 0.0;
    double admission_seconds = 0.0;
    double two_hop_seconds = 0.0;
};

__global__ void basc_priority_kernel(
    std::int64_t n, const std::int64_t* offsets, std::uint32_t seed,
    float alpha, float* priority) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const auto degree = offsets[v + 1] - offsets[v];
    const auto random_bits = mix32(
        static_cast<std::uint32_t>(v) ^ seed ^ 0x9e3779b9U);
    const float random01 = (static_cast<float>(random_bits) + 1.0f) /
                           4294967297.0f;
    priority[v] = random01 / powf(static_cast<float>(degree) + 1.0f, alpha);
}

__global__ void basc_anchor_election_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const float* priority,
    std::uint8_t* anchors) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    bool local_max = true;
    for (auto e = offsets[v]; e < offsets[v + 1]; ++e) {
        const auto u = neighbors[e];
        if (priority[u] > priority[v] ||
            (priority[u] == priority[v] && u < v)) {
            local_max = false;
            break;
        }
    }
    anchors[v] = local_max ? 1 : 0;
}

__device__ __forceinline__ void basc_insert_candidate(
    int k, std::int32_t anchor, float affinity,
    std::int32_t* candidates, float* affinities) {
    int existing = -1;
    for (int j = 0; j < k; ++j) {
        if (candidates[j] == anchor) {
            existing = j;
            break;
        }
    }
    if (existing >= 0) {
        affinities[existing] += affinity;
        return;
    }
    int position = k;
    for (int j = 0; j < k; ++j) {
        if (affinity > affinities[j] ||
            (affinity == affinities[j] && anchor < candidates[j])) {
            position = j;
            break;
        }
    }
    if (position >= k) return;
    for (int j = k - 1; j > position; --j) {
        candidates[j] = candidates[j - 1];
        affinities[j] = affinities[j - 1];
    }
    candidates[position] = anchor;
    affinities[position] = affinity;
}

__global__ void basc_anchor_candidates_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint64_t* edge_weights,
    const std::int64_t* degrees, const std::uint8_t* anchors, int k,
    std::int32_t* candidates, float* affinities,
    std::uint8_t* candidate_counts) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    auto* my_candidates = candidates + v * BASC_MAX_K;
    auto* my_affinities = affinities + v * BASC_MAX_K;
    for (int j = 0; j < BASC_MAX_K; ++j) {
        my_candidates[j] = BASC_INVALID;
        my_affinities[j] = 0.0f;
    }
    if (anchors[v]) {
        my_candidates[0] = static_cast<std::int32_t>(v);
        my_affinities[0] = 1.0f;
        candidate_counts[v] = 1;
        return;
    }
    for (auto e = offsets[v]; e < offsets[v + 1]; ++e) {
        const auto u = neighbors[e];
        if (!anchors[u]) continue;
        const float denominator = sqrtf(static_cast<float>(degrees[u]) + 1.0f);
        const float affinity = static_cast<float>(edge_weights[e]) / denominator;
        basc_insert_candidate(k, u, affinity, my_candidates, my_affinities);
    }
    std::uint8_t valid = 0;
    for (int j = 0; j < k; ++j) {
        valid += my_candidates[j] >= 0 ? 1 : 0;
    }
    candidate_counts[v] = valid;
}

__global__ void basc_support_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint64_t* edge_weights,
    const std::uint8_t* anchors, int k, float lambda,
    const std::int32_t* candidates, const float* affinities,
    std::int32_t* proposals, float* confidence, std::uint8_t* buckets) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (anchors[v]) {
        proposals[v] = static_cast<std::int32_t>(v);
        confidence[v] = 1.0f;
        buckets[v] = 3;
        return;
    }
    const auto* my_candidates = candidates + v * BASC_MAX_K;
    const auto* my_affinities = affinities + v * BASC_MAX_K;
    std::uint64_t support[BASC_MAX_K] = {};
    std::uint64_t degree_weight = 0;
    for (auto e = offsets[v]; e < offsets[v + 1]; ++e) {
        const auto u = neighbors[e];
        const auto weight = edge_weights[e];
        degree_weight += weight;
        const auto* neighbor_candidates = candidates +
            static_cast<std::int64_t>(u) * BASC_MAX_K;
        for (int j = 0; j < k; ++j) {
            if (my_candidates[j] >= 0 &&
                neighbor_candidates[0] == my_candidates[j]) {
                support[j] += weight;
            }
        }
    }
    float affinity_sum = 0.0f;
    for (int j = 0; j < k; ++j) affinity_sum += my_affinities[j];
    int best = -1;
    float best_score = -1.0f;
    for (int j = 0; j < k; ++j) {
        if (my_candidates[j] < 0) continue;
        const float direct = my_affinities[j] / (affinity_sum + 1.0e-20f);
        const float support_ratio = degree_weight == 0
            ? 0.0f : static_cast<float>(support[j]) /
                         static_cast<float>(degree_weight);
        const float score = lambda * direct + (1.0f - lambda) * support_ratio;
        if (best < 0 || score > best_score ||
            (score == best_score && my_candidates[j] < my_candidates[best])) {
            best = j;
            best_score = score;
        }
    }
    if (best < 0) {
        proposals[v] = BASC_INVALID;
        confidence[v] = 0.0f;
        buckets[v] = 0;
    } else {
        proposals[v] = my_candidates[best];
        confidence[v] = best_score;
        buckets[v] = static_cast<std::uint8_t>(min(3, max(0, static_cast<int>(best_score * 4.0f))));
    }
}

__global__ void basc_initialize_aggregates_kernel(
    std::int64_t n, const std::uint64_t* vertex_weights,
    const std::uint8_t* anchors, std::int32_t* aggregates,
    unsigned long long* aggregate_weights) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (anchors[v]) {
        aggregates[v] = static_cast<std::int32_t>(v);
        aggregate_weights[v] = static_cast<unsigned long long>(vertex_weights[v]);
    } else {
        aggregates[v] = BASC_INVALID;
        aggregate_weights[v] = 0ULL;
    }
}

__global__ void basc_admit_bucket_kernel(
    std::int64_t n, int bucket, std::uint64_t capacity,
    const std::uint64_t* vertex_weights, const std::int32_t* proposals,
    const std::uint8_t* buckets, std::int32_t* aggregates,
    unsigned long long* aggregate_weights, unsigned long long* accepted,
    unsigned long long* capacity_rejected) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n || buckets[v] != bucket || aggregates[v] != BASC_INVALID) return;
    const auto target = proposals[v];
    if (target < 0) return;
    const auto weight = static_cast<unsigned long long>(vertex_weights[v]);
    auto* slot = &aggregate_weights[target];
    auto old = atomicAdd(slot, 0ULL);
    while (old <= capacity && weight <= capacity - old) {
        const auto previous = atomicCAS(slot, old, old + weight);
        if (previous == old) {
            aggregates[v] = target;
            atomicAdd(accepted, 1ULL);
            return;
        }
        old = previous;
    }
    atomicAdd(capacity_rejected, 1ULL);
}

__device__ __forceinline__ void basc_insert_support_candidate(
    std::int32_t aggregate, std::uint64_t support,
    std::int32_t& first, std::uint64_t& first_support,
    std::int32_t& second, std::uint64_t& second_support) {
    if (aggregate < 0) return;
    if (aggregate == first) {
        first_support += support;
    } else if (aggregate == second) {
        second_support += support;
    } else if (first < 0) {
        first = aggregate;
        first_support = support;
    } else if (second < 0) {
        second = aggregate;
        second_support = support;
    } else if (support > second_support ||
               (support == second_support && aggregate < second)) {
        second = aggregate;
        second_support = support;
    }
    if (second_support > first_support ||
        (second_support == first_support && second >= 0 && first >= 0 && second < first)) {
        const auto tmp_id = first;
        const auto tmp_support = first_support;
        first = second;
        first_support = second_support;
        second = tmp_id;
        second_support = tmp_support;
    }
}

__global__ void basc_expansion_kernel(
    std::int64_t n, float threshold, std::uint64_t capacity,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::uint64_t* edge_weights, const std::uint64_t* vertex_weights,
    std::int32_t* aggregates, unsigned long long* aggregate_weights,
    unsigned long long* accepted, unsigned long long* capacity_rejected,
    unsigned long long* eligible) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n || aggregates[v] != BASC_INVALID) return;
    std::int32_t first = BASC_INVALID;
    std::int32_t second = BASC_INVALID;
    std::uint64_t first_support = 0;
    std::uint64_t second_support = 0;
    std::uint64_t total = 0;
    for (auto e = offsets[v]; e < offsets[v + 1]; ++e) {
        const auto weight = edge_weights[e];
        total += weight;
        basc_insert_support_candidate(
            aggregates[neighbors[e]], weight, first, first_support,
            second, second_support);
    }
    if (first < 0 || total == 0 ||
        static_cast<float>(first_support) < threshold * static_cast<float>(total)) return;
    atomicAdd(eligible, 1ULL);
    const auto weight = static_cast<unsigned long long>(vertex_weights[v]);
    auto* slot = &aggregate_weights[first];
    auto old = atomicAdd(slot, 0ULL);
    while (old <= capacity && weight <= capacity - old) {
        const auto previous = atomicCAS(slot, old, old + weight);
        if (previous == old) {
            aggregates[v] = first;
            atomicAdd(accepted, 1ULL);
            return;
        }
        old = previous;
    }
    atomicAdd(capacity_rejected, 1ULL);
}

__global__ void basc_singleton_kernel(
    std::int64_t n, const std::uint64_t* vertex_weights,
    std::int32_t* aggregates, unsigned long long* aggregate_weights,
    unsigned long long* singletons) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n || aggregates[v] != BASC_INVALID) return;
    aggregates[v] = static_cast<std::int32_t>(v);
    aggregate_weights[v] = static_cast<unsigned long long>(vertex_weights[v]);
    atomicAdd(singletons, 1ULL);
}

// Performance path: one warp owns one vertex while scanning CSR.  The
// existing basc_* kernels above intentionally remain as a correctness
// baseline; these kernels avoid one-thread-per-vertex serial edge walks.
__global__ void basc_inverse_sqrt_degree_kernel(
    std::int64_t n, const std::int64_t* offsets, float* inverse_degree) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const auto degree = offsets[v + 1] - offsets[v];
    inverse_degree[v] = rsqrtf(static_cast<float>(degree) + 1.0f);
}

__global__ void basc_warp_anchor_election_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const float* priority,
    std::uint8_t* anchors) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + (threadIdx.x / BASC_WARP_SIZE);
    if (warp >= n) return;
    const unsigned mask = __activemask();
    const auto begin = offsets[warp];
    const auto end = offsets[warp + 1];
    bool dominated = false;
    for (auto e = begin + lane; e < end; e += BASC_WARP_SIZE) {
        const auto u = neighbors[e];
        if (priority[u] > priority[warp] ||
            (priority[u] == priority[warp] && u < warp)) {
            dominated = true;
        }
    }
    if (lane == 0) {
        bool any_dominated = false;
        for (int other = 0; other < BASC_WARP_SIZE; ++other) {
            any_dominated = any_dominated ||
                (__shfl_sync(mask, dominated, other) != 0);
        }
        anchors[warp] = any_dominated ? 0 : 1;
    }
}

__global__ void basc_warp_anchor_candidates_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint64_t* edge_weights,
    const float* inv_sqrt_degree, const std::uint8_t* anchors, int k,
    std::int32_t* candidates, float* affinities,
    std::uint8_t* candidate_counts) {
    __shared__ std::int32_t shared_candidates[
        BASC_WARPS_PER_BLOCK * BASC_WARP_SIZE * BASC_MAX_K];
    __shared__ float shared_affinities[
        BASC_WARPS_PER_BLOCK * BASC_WARP_SIZE * BASC_MAX_K];
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const int local_warp = threadIdx.x / BASC_WARP_SIZE;
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + local_warp;
    if (warp >= n) return;
    const unsigned mask = __activemask();
    std::int32_t local_candidates[BASC_MAX_K];
    float local_affinities[BASC_MAX_K];
    for (int j = 0; j < BASC_MAX_K; ++j) {
        local_candidates[j] = BASC_INVALID;
        local_affinities[j] = 0.0f;
    }
    if (!anchors[warp]) {
        for (auto e = offsets[warp] + lane; e < offsets[warp + 1]; e += BASC_WARP_SIZE) {
            const auto u = neighbors[e];
            if (!anchors[u]) continue;
            const auto affinity = static_cast<float>(edge_weights[e]) * inv_sqrt_degree[u];
            basc_insert_candidate(k, u, affinity, local_candidates, local_affinities);
        }
    } else if (lane == 0) {
        local_candidates[0] = static_cast<std::int32_t>(warp);
        local_affinities[0] = 1.0f;
    }

    const auto shared_base = local_warp * BASC_WARP_SIZE * BASC_MAX_K + lane * BASC_MAX_K;
    for (int j = 0; j < BASC_MAX_K; ++j) {
        shared_candidates[shared_base + j] = local_candidates[j];
        shared_affinities[shared_base + j] = local_affinities[j];
    }
    __syncwarp(mask);
    if (lane == 0) {
        std::int32_t merged_candidates[BASC_MAX_K];
        float merged_affinities[BASC_MAX_K];
        for (int j = 0; j < BASC_MAX_K; ++j) {
            merged_candidates[j] = BASC_INVALID;
            merged_affinities[j] = 0.0f;
        }
        for (int other = 0; other < BASC_WARP_SIZE; ++other) {
            const auto base = local_warp * BASC_WARP_SIZE * BASC_MAX_K +
                              other * BASC_MAX_K;
            for (int j = 0; j < k; ++j) {
                if (shared_candidates[base + j] >= 0) {
                    basc_insert_candidate(
                        k, shared_candidates[base + j], shared_affinities[base + j],
                        merged_candidates, merged_affinities);
                }
            }
        }
        auto* output_candidates = candidates + warp * BASC_MAX_K;
        auto* output_affinities = affinities + warp * BASC_MAX_K;
        std::uint8_t valid = 0;
        for (int j = 0; j < BASC_MAX_K; ++j) {
            output_candidates[j] = merged_candidates[j];
            output_affinities[j] = merged_affinities[j];
            if (j < k && merged_candidates[j] >= 0) ++valid;
        }
        candidate_counts[warp] = valid;
    }
}

__global__ void basc_warp_support_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint64_t* edge_weights,
    const std::uint8_t* anchors, int k, float lambda,
    const std::int32_t* candidates, const float* affinities,
    std::int32_t* proposals, float* confidence, std::uint8_t* buckets) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + (threadIdx.x / BASC_WARP_SIZE);
    if (warp >= n) return;
    const unsigned mask = __activemask();
    if (anchors[warp]) {
        if (lane == 0) {
            proposals[warp] = static_cast<std::int32_t>(warp);
            confidence[warp] = 1.0f;
            buckets[warp] = 3;
        }
        return;
    }
    const auto* my_candidates = candidates + warp * BASC_MAX_K;
    const auto* my_affinities = affinities + warp * BASC_MAX_K;
    std::uint64_t support[BASC_MAX_K] = {};
    std::uint64_t degree_weight = 0;
    for (auto e = offsets[warp] + lane; e < offsets[warp + 1]; e += BASC_WARP_SIZE) {
        const auto u = neighbors[e];
        const auto weight = edge_weights[e];
        degree_weight += weight;
        const auto* neighbor_candidates = candidates +
            static_cast<std::int64_t>(u) * BASC_MAX_K;
        for (int j = 0; j < k; ++j) {
            if (my_candidates[j] < 0) continue;
            for (int q = 0; q < k; ++q) {
                if (neighbor_candidates[q] == my_candidates[j]) {
                    support[j] += weight;
                    break;
                }
            }
        }
    }
    for (int delta = BASC_WARP_SIZE / 2; delta > 0; delta >>= 1) {
        degree_weight += __shfl_down_sync(mask, degree_weight, delta);
        for (int j = 0; j < k; ++j) {
            support[j] += __shfl_down_sync(mask, support[j], delta);
        }
    }
    if (lane == 0) {
        float affinity_sum = 0.0f;
        for (int j = 0; j < k; ++j) affinity_sum += my_affinities[j];
        int best = -1;
        float best_score = -1.0f;
        for (int j = 0; j < k; ++j) {
            if (my_candidates[j] < 0) continue;
            const float direct = my_affinities[j] / (affinity_sum + 1.0e-20f);
            const float support_ratio = degree_weight == 0
                ? 0.0f : static_cast<float>(support[j]) /
                           static_cast<float>(degree_weight);
            const float score = lambda * direct +
                                (1.0f - lambda) * support_ratio;
            if (best < 0 || score > best_score ||
                (score == best_score && my_candidates[j] < my_candidates[best])) {
                best = j;
                best_score = score;
            }
        }
        if (best < 0) {
            proposals[warp] = BASC_INVALID;
            confidence[warp] = 0.0f;
            buckets[warp] = 0;
        } else {
            proposals[warp] = my_candidates[best];
            confidence[warp] = best_score;
            buckets[warp] = static_cast<std::uint8_t>(min(
                3, max(0, static_cast<int>(best_score * 4.0f))));
        }
    }
}

__global__ void basc_warp_expansion_propose_kernel(
    std::int64_t n, float threshold, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint64_t* edge_weights,
    const std::int32_t* aggregates, std::int32_t* targets,
    std::uint8_t* eligible, unsigned long long* eligible_count) {
    __shared__ std::int32_t shared_aggregates[
        BASC_WARPS_PER_BLOCK * BASC_WARP_SIZE * 2];
    __shared__ std::uint64_t shared_support[
        BASC_WARPS_PER_BLOCK * BASC_WARP_SIZE * 2];
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const int local_warp = threadIdx.x / BASC_WARP_SIZE;
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + local_warp;
    if (warp >= n) return;
    const unsigned mask = __activemask();
    if (aggregates[warp] != BASC_INVALID) return;

    std::int32_t first = BASC_INVALID;
    std::int32_t second = BASC_INVALID;
    std::uint64_t first_support = 0;
    std::uint64_t second_support = 0;
    std::uint64_t total = 0;
    for (auto e = offsets[warp] + lane; e < offsets[warp + 1]; e += BASC_WARP_SIZE) {
        const auto weight = edge_weights[e];
        total += weight;
        basc_insert_support_candidate(
            aggregates[neighbors[e]], weight, first, first_support,
            second, second_support);
    }
    const auto base = local_warp * BASC_WARP_SIZE * 2 + lane * 2;
    shared_aggregates[base] = first;
    shared_aggregates[base + 1] = second;
    shared_support[base] = first_support;
    shared_support[base + 1] = second_support;
    __syncwarp(mask);
    for (int delta = BASC_WARP_SIZE / 2; delta > 0; delta >>= 1) {
        total += __shfl_down_sync(mask, total, delta);
    }
    if (lane == 0) {
        first = BASC_INVALID;
        second = BASC_INVALID;
        first_support = 0;
        second_support = 0;
        for (int other = 0; other < BASC_WARP_SIZE; ++other) {
            const auto other_base = local_warp * BASC_WARP_SIZE * 2 + other * 2;
            basc_insert_support_candidate(
                shared_aggregates[other_base], shared_support[other_base],
                first, first_support, second, second_support);
            basc_insert_support_candidate(
                shared_aggregates[other_base + 1], shared_support[other_base + 1],
                first, first_support, second, second_support);
        }
        targets[warp] = BASC_INVALID;
        eligible[warp] = 0;
        if (first >= 0 && total > 0 &&
            static_cast<float>(first_support) >= threshold * static_cast<float>(total)) {
            targets[warp] = first;
            eligible[warp] = 1;
            atomicAdd(eligible_count, 1ULL);
        }
    }
}

__global__ void basc_expansion_commit_kernel(
    std::int64_t n, std::uint64_t capacity,
    const std::uint64_t* vertex_weights, const std::int32_t* targets,
    std::int32_t* aggregates, unsigned long long* aggregate_weights,
    unsigned long long* accepted, unsigned long long* rejected) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n || targets[v] < 0 || aggregates[v] != BASC_INVALID) return;
    const auto target = targets[v];
    const auto weight = static_cast<unsigned long long>(vertex_weights[v]);
    auto* slot = &aggregate_weights[target];
    auto old = atomicAdd(slot, 0ULL);
    while (old <= capacity && weight <= capacity - old) {
        const auto previous = atomicCAS(slot, old, old + weight);
        if (previous == old) {
            aggregates[v] = target;
            atomicAdd(accepted, 1ULL);
            return;
        }
        old = previous;
    }
    atomicAdd(rejected, 1ULL);
}

__global__ void basc_mark_roots_kernel(
    std::int64_t n, const std::int32_t* aggregates, std::int32_t* root_flags) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    root_flags[v] = aggregates[v] == v ? 1 : 0;
}

__global__ void basc_compact_map_kernel(
    std::int64_t n, const std::int32_t* aggregates,
    const std::int32_t* root_ids, std::int32_t* map) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    map[v] = root_ids[aggregates[v]];
}

__global__ void basc_coarse_vertex_weights_kernel(
    std::int64_t n, const std::uint64_t* vertex_weights,
    const std::int32_t* map, std::uint64_t* coarse_weights) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    atomicAdd(reinterpret_cast<unsigned long long*>(&coarse_weights[map[v]]),
              static_cast<unsigned long long>(vertex_weights[v]));
}

__global__ void basc_fill_contraction_keys_warp_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint64_t* edge_weights,
    const std::int32_t* map, std::uint64_t* keys, std::uint64_t* values) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + (threadIdx.x / BASC_WARP_SIZE);
    if (warp >= n) return;
    const auto source = static_cast<std::uint32_t>(map[warp]);
    for (auto e = offsets[warp] + lane; e < offsets[warp + 1]; e += BASC_WARP_SIZE) {
        const auto target = static_cast<std::uint32_t>(map[neighbors[e]]);
        keys[e] = source == target
            ? BASC_INVALID_KEY
            : (static_cast<std::uint64_t>(source) << 32) | target;
        values[e] = edge_weights[e];
    }
}

__global__ void basc_count_coarse_rows_kernel(
    std::int64_t count, const std::uint64_t* keys,
    std::int64_t coarse_vertices, std::int64_t* row_counts) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto source = static_cast<std::uint32_t>(keys[i] >> 32);
    if (source >= static_cast<std::uint32_t>(coarse_vertices)) return;
    // Store the row count at its source index.  The extra final zero entry is
    // consumed by the exclusive scan below to produce offsets[nc].
    atomicAdd(reinterpret_cast<unsigned long long*>(&row_counts[source]), 1ULL);
}

__global__ void basc_write_coarse_edges_kernel(
    std::int64_t count, const std::uint64_t* keys,
    const std::uint64_t* values, std::int32_t* neighbors,
    std::uint64_t* edge_weights) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    neighbors[i] = static_cast<std::int32_t>(keys[i]);
    edge_weights[i] = values[i];
}

constexpr int FRONTIER_LABEL_K = 4;

__global__ void frontier_degree_kernel(
    std::int64_t n, const std::int64_t* offsets, std::int64_t* degrees) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v < n) degrees[v] = offsets[v + 1] - offsets[v];
}

__global__ void frontier_local_degree_max_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::int64_t* degrees,
    std::uint8_t* local_maximum) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + threadIdx.x / BASC_WARP_SIZE;
    if (warp >= n) return;
    const unsigned mask = __activemask();
    bool dominated = false;
    for (auto e = offsets[warp] + lane; e < offsets[warp + 1]; e += BASC_WARP_SIZE) {
        const auto u = neighbors[e];
        if (degrees[u] > degrees[warp] ||
            (degrees[u] == degrees[warp] && u < warp)) {
            dominated = true;
        }
    }
    const bool any = __any_sync(mask, dominated);
    if (lane == 0) local_maximum[warp] = any ? 0 : 1;
}

__global__ void frontier_hot_scores_kernel(
    std::int64_t n, const std::int64_t* degrees,
    const std::uint8_t* local_maximum, double inverse_log_max_degree,
    float* scores) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    const auto degree = degrees[v];
    const float hotness = degree == 0 ? 0.0f : static_cast<float>(
        log1p(static_cast<double>(degree)) * inverse_log_max_degree);
    // Local maxima always precede fallback vertices.  Isolated vertices are
    // left for the cold pool unless there are not enough non-isolated seeds.
    scores[v] = degree > 0
        ? (local_maximum[v] ? 4.0f + hotness : 2.0f + hotness)
        : 0.0f;
}

__global__ void frontier_initialize_seeds_kernel(
    std::int64_t count, const std::int32_t* seed_vertices,
    const std::uint64_t* vertex_weights, std::int32_t* labels,
    std::uint8_t* seed_mask, unsigned long long* cluster_weights) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto v = seed_vertices[i];
    labels[v] = v;
    seed_mask[v] = 1;
    cluster_weights[v] = static_cast<unsigned long long>(vertex_weights[v]);
}

__global__ void frontier_cover_hot_kernel(
    std::int64_t count, const std::int32_t* hot_vertices,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    std::uint8_t* covered) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + threadIdx.x / BASC_WARP_SIZE;
    if (warp >= count) return;
    const auto v = hot_vertices[warp];
    if (lane == 0) covered[v] = 1;
    for (auto e = offsets[v] + lane; e < offsets[v + 1]; e += BASC_WARP_SIZE) {
        covered[neighbors[e]] = 1;
    }
}

__global__ void frontier_cold_scores_kernel(
    std::int64_t n, const std::int64_t* degrees,
    const std::uint8_t* covered, const std::int32_t* labels,
    double inverse_log_max_degree, std::uint32_t seed, float* scores) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (labels[v] >= 0) {
        scores[v] = -3.402823466e+38F;
        return;
    }
    const auto degree = degrees[v];
    const float hotness = degree == 0 ? 0.0f : static_cast<float>(
        log1p(static_cast<double>(degree)) * inverse_log_max_degree);
    const auto bits = mix32(static_cast<std::uint32_t>(v) ^ seed ^ 0x85ebca6bU);
    const float random_tie = static_cast<float>(bits) / 4294967295.0f;
    scores[v] = (1.0f - hotness) + 0.1f * random_tie;
}

__global__ void frontier_cold_local_max_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint8_t* covered,
    const std::int32_t* labels, const float* scores,
    std::uint8_t* local_maximum) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + threadIdx.x / BASC_WARP_SIZE;
    if (warp >= n) return;
    const unsigned mask = __activemask();
    if (labels[warp] >= 0 || covered[warp]) {
        if (lane == 0) local_maximum[warp] = 0;
        return;
    }
    bool dominated = false;
    for (auto e = offsets[warp] + lane; e < offsets[warp + 1]; e += BASC_WARP_SIZE) {
        const auto u = neighbors[e];
        if (labels[u] >= 0 || covered[u]) continue;
        if (scores[u] > scores[warp] ||
            (scores[u] == scores[warp] && u < warp)) dominated = true;
    }
    const bool any = __any_sync(mask, dominated);
    if (lane == 0) local_maximum[warp] = any ? 0 : 1;
}

__global__ void frontier_cold_rank_kernel(
    std::int64_t n, const std::int64_t* degrees,
    const std::uint8_t* covered,
    const std::uint8_t* local_maximum, const std::int32_t* labels,
    float* scores) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (labels[v] >= 0) {
        scores[v] = -3.402823466e+38F;
    } else if (!covered[v] && degrees[v] > 1 && local_maximum[v]) {
        scores[v] += 14.0f;
    } else if (!covered[v] && local_maximum[v]) {
        scores[v] += 12.0f;
    } else if (!covered[v] && degrees[v] > 1) {
        scores[v] += 8.0f;
    } else if (!covered[v]) {
        scores[v] += 4.0f;
    } else {
        scores[v] += 2.0f;
    }
}

__global__ void frontier_mark_neighbors_kernel(
    std::int64_t frontier_count, const std::int32_t* frontier,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::int32_t* labels, std::int32_t epoch,
    std::int32_t* active_epoch, std::int32_t* active,
    unsigned long long* active_count) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + threadIdx.x / BASC_WARP_SIZE;
    if (warp >= frontier_count) return;
    const auto v = frontier[warp];
    for (auto e = offsets[v] + lane; e < offsets[v + 1]; e += BASC_WARP_SIZE) {
        const auto u = neighbors[e];
        if (labels[u] >= 0) continue;
        const auto previous = atomicExch(
            reinterpret_cast<int*>(&active_epoch[u]), static_cast<int>(epoch));
        if (previous != epoch) {
            const auto slot = atomicAdd(active_count, 1ULL);
            active[slot] = u;
        }
    }
}

__device__ __forceinline__ void frontier_insert_label_support(
    std::int32_t label, std::uint64_t weight,
    std::int32_t* labels, std::uint64_t* support) {
    if (label < 0) return;
    int empty = -1;
    int weakest = 0;
    for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
        if (labels[j] == label) {
            support[j] += weight;
            return;
        }
        if (labels[j] < 0 && empty < 0) empty = j;
        if (support[j] < support[weakest] ||
            (support[j] == support[weakest] && labels[j] > labels[weakest])) {
            weakest = j;
        }
    }
    const int slot = empty >= 0 ? empty : weakest;
    if (empty < 0 && (weight < support[slot] ||
        (weight == support[slot] && label > labels[slot]))) return;
    labels[slot] = label;
    support[slot] = weight;
}

__global__ void frontier_growth_propose_kernel(
    std::int64_t active_count, const std::int32_t* active,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::uint64_t* edge_weights, const std::uint64_t* vertex_weights,
    const std::int32_t* labels, const unsigned long long* cluster_weights,
    std::uint64_t capacity, float confidence_threshold,
    std::uint8_t max_postpone, std::uint8_t* postpone,
    std::int32_t* proposals, float* proposal_scores,
    unsigned long long* proposed_count, unsigned long long* postponed_count) {
    __shared__ std::int32_t shared_labels[
        BASC_WARPS_PER_BLOCK * BASC_WARP_SIZE * FRONTIER_LABEL_K];
    __shared__ std::uint64_t shared_support[
        BASC_WARPS_PER_BLOCK * BASC_WARP_SIZE * FRONTIER_LABEL_K];
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const int local_warp = threadIdx.x / BASC_WARP_SIZE;
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + local_warp;
    if (warp >= active_count) return;
    const unsigned mask = __activemask();
    const auto v = active[warp];
    std::int32_t local_labels[FRONTIER_LABEL_K];
    std::uint64_t local_support[FRONTIER_LABEL_K];
    for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
        local_labels[j] = BASC_INVALID;
        local_support[j] = 0;
    }
    std::uint64_t total_labeled = 0;
    if (labels[v] < 0) {
        for (auto e = offsets[v] + lane; e < offsets[v + 1]; e += BASC_WARP_SIZE) {
            const auto label = labels[neighbors[e]];
            if (label < 0) continue;
            const auto weight = edge_weights[e];
            total_labeled += weight;
            frontier_insert_label_support(
                label, weight, local_labels, local_support);
        }
    }
    const auto base = local_warp * BASC_WARP_SIZE * FRONTIER_LABEL_K +
                      lane * FRONTIER_LABEL_K;
    for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
        shared_labels[base + j] = local_labels[j];
        shared_support[base + j] = local_support[j];
    }
    __syncwarp(mask);
    for (int delta = BASC_WARP_SIZE / 2; delta > 0; delta >>= 1) {
        total_labeled += __shfl_down_sync(mask, total_labeled, delta);
    }
    if (lane != 0) return;

    proposals[v] = BASC_INVALID;
    proposal_scores[v] = -1.0f;
    if (labels[v] >= 0) return;
    std::int32_t merged_labels[FRONTIER_LABEL_K];
    std::uint64_t merged_support[FRONTIER_LABEL_K];
    for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
        merged_labels[j] = BASC_INVALID;
        merged_support[j] = 0;
    }
    for (int other = 0; other < BASC_WARP_SIZE; ++other) {
        const auto other_base = local_warp * BASC_WARP_SIZE * FRONTIER_LABEL_K +
                                other * FRONTIER_LABEL_K;
        for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
            frontier_insert_label_support(
                shared_labels[other_base + j], shared_support[other_base + j],
                merged_labels, merged_support);
        }
    }
    int best = -1;
    float best_score = -1.0f;
    const auto own_weight = vertex_weights[v];
    for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
        const auto label = merged_labels[j];
        if (label < 0) continue;
        const auto used = static_cast<std::uint64_t>(cluster_weights[label]);
        if (used > capacity || own_weight > capacity - used) continue;
        const float remaining = capacity == 0 ? 0.0f :
            static_cast<float>(capacity - used) / static_cast<float>(capacity);
        const float score = static_cast<float>(merged_support[j]) * remaining;
        if (best < 0 || score > best_score ||
            (score == best_score && label < merged_labels[best])) {
            best = j;
            best_score = score;
        }
    }
    if (best < 0) return;
    const float confidence = total_labeled == 0 ? 0.0f :
        static_cast<float>(merged_support[best]) /
        static_cast<float>(total_labeled);
    if (confidence < confidence_threshold && postpone[v] < max_postpone) {
        ++postpone[v];
        atomicAdd(postponed_count, 1ULL);
        return;
    }
    proposals[v] = merged_labels[best];
    proposal_scores[v] = best_score;
    atomicAdd(proposed_count, 1ULL);
}

struct FrontierProposalOrder {
    const std::int32_t* proposals;
    const float* scores;
    __host__ __device__ bool operator()(std::int32_t a, std::int32_t b) const {
        const auto pa = proposals[a];
        const auto pb = proposals[b];
        if ((pa >= 0) != (pb >= 0)) return pa >= 0;
        if (pa != pb) return pa < pb;
        if (scores[a] != scores[b]) return scores[a] > scores[b];
        return a < b;
    }
};

struct FrontierHasProposal {
    const std::int32_t* proposals;
    __host__ __device__ bool operator()(std::int32_t v) const {
        return proposals[v] >= 0;
    }
};

__global__ void frontier_ordered_proposal_data_kernel(
    std::int64_t count, const std::int32_t* order,
    const std::int32_t* proposals, const std::uint64_t* vertex_weights,
    std::int32_t* targets, std::uint64_t* weights) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto v = order[i];
    targets[i] = proposals[v];
    weights[i] = vertex_weights[v];
}

__global__ void frontier_commit_prefix_kernel(
    std::int64_t count, const std::int32_t* order,
    const std::int32_t* targets, const std::uint64_t* weights,
    const std::uint64_t* prefix_weights,
    const unsigned long long* base_cluster_weights, std::uint64_t capacity,
    std::int32_t* labels, unsigned long long* cluster_weights,
    std::int32_t* accepted_frontier, unsigned long long* accepted_count,
    unsigned long long* rejected_count) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto target = targets[i];
    const auto base = static_cast<std::uint64_t>(base_cluster_weights[target]);
    const auto prefix = prefix_weights[i];
    if (base <= capacity && prefix <= capacity - base) {
        const auto v = order[i];
        labels[v] = target;
        atomicAdd(&cluster_weights[target],
                  static_cast<unsigned long long>(weights[i]));
        const auto slot = atomicAdd(accepted_count, 1ULL);
        accepted_frontier[slot] = v;
    } else {
        atomicAdd(rejected_count, 1ULL);
    }
}

__global__ void frontier_carry_unassigned_kernel(
    std::int64_t count, const std::int32_t* vertices,
    const std::int32_t* labels, std::int32_t epoch,
    std::int32_t* active_epoch, std::int32_t* output,
    unsigned long long* output_count) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto v = vertices[i];
    if (labels[v] >= 0) return;
    const auto previous = atomicExch(
        reinterpret_cast<int*>(&active_epoch[v]), static_cast<int>(epoch));
    if (previous != epoch) {
        const auto slot = atomicAdd(output_count, 1ULL);
        output[slot] = v;
    }
}

__global__ void frontier_emergency_mark_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const float* priority,
    const std::int32_t* labels, std::int32_t* flags) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + threadIdx.x / BASC_WARP_SIZE;
    if (warp >= n) return;
    const unsigned mask = __activemask();
    if (labels[warp] >= 0) {
        if (lane == 0) flags[warp] = 0;
        return;
    }
    bool dominated = false;
    for (auto e = offsets[warp] + lane; e < offsets[warp + 1]; e += BASC_WARP_SIZE) {
        const auto u = neighbors[e];
        if (labels[u] >= 0) continue;
        if (priority[u] > priority[warp] ||
            (priority[u] == priority[warp] && u < warp)) dominated = true;
    }
    const bool any = __any_sync(mask, dominated);
    if (lane == 0) flags[warp] = any ? 0 : 1;
}

__global__ void frontier_compact_flags_kernel(
    std::int64_t n, const std::int32_t* flags,
    const std::int32_t* positions, std::int32_t* output) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v < n && flags[v]) output[positions[v]] = static_cast<std::int32_t>(v);
}

__global__ void frontier_boundary_propose_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint64_t* edge_weights,
    const std::uint64_t* vertex_weights, const std::int32_t* labels,
    const std::uint8_t* seed_mask,
    const unsigned long long* cluster_weights, std::uint64_t capacity,
    std::int32_t* proposals, float* proposal_scores,
    std::uint64_t* gains) {
    __shared__ std::int32_t shared_labels[
        BASC_WARPS_PER_BLOCK * BASC_WARP_SIZE * FRONTIER_LABEL_K];
    __shared__ std::uint64_t shared_support[
        BASC_WARPS_PER_BLOCK * BASC_WARP_SIZE * FRONTIER_LABEL_K];
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const int local_warp = threadIdx.x / BASC_WARP_SIZE;
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + local_warp;
    if (warp >= n) return;
    const unsigned mask = __activemask();
    const auto current = labels[warp];
    std::int32_t local_labels[FRONTIER_LABEL_K];
    std::uint64_t local_support[FRONTIER_LABEL_K];
    for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
        local_labels[j] = BASC_INVALID;
        local_support[j] = 0;
    }
    std::uint64_t current_support = 0;
    if (!seed_mask[warp]) {
        for (auto e = offsets[warp] + lane; e < offsets[warp + 1]; e += BASC_WARP_SIZE) {
            const auto neighbor = neighbors[e];
            if (neighbor == warp) continue;  // A self-loop is unchanged by relabeling.
            const auto neighbor_label = labels[neighbor];
            const auto weight = edge_weights[e];
            if (neighbor_label == current) current_support += weight;
            frontier_insert_label_support(
                neighbor_label, weight, local_labels, local_support);
        }
    }
    const auto base = local_warp * BASC_WARP_SIZE * FRONTIER_LABEL_K +
                      lane * FRONTIER_LABEL_K;
    for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
        shared_labels[base + j] = local_labels[j];
        shared_support[base + j] = local_support[j];
    }
    __syncwarp(mask);
    for (int delta = BASC_WARP_SIZE / 2; delta > 0; delta >>= 1) {
        current_support += __shfl_down_sync(mask, current_support, delta);
    }
    if (lane != 0) return;
    proposals[warp] = BASC_INVALID;
    proposal_scores[warp] = -1.0f;
    gains[warp] = 0;
    if (seed_mask[warp]) return;

    std::int32_t merged_labels[FRONTIER_LABEL_K];
    std::uint64_t merged_support[FRONTIER_LABEL_K];
    for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
        merged_labels[j] = BASC_INVALID;
        merged_support[j] = 0;
    }
    for (int other = 0; other < BASC_WARP_SIZE; ++other) {
        const auto other_base = local_warp * BASC_WARP_SIZE * FRONTIER_LABEL_K +
                                other * FRONTIER_LABEL_K;
        for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
            frontier_insert_label_support(
                shared_labels[other_base + j], shared_support[other_base + j],
                merged_labels, merged_support);
        }
    }
    int best = -1;
    std::uint64_t best_gain = 0;
    const auto own_weight = vertex_weights[warp];
    for (int j = 0; j < FRONTIER_LABEL_K; ++j) {
        const auto target = merged_labels[j];
        if (target < 0 || target == current || merged_support[j] <= current_support) continue;
        const auto used = static_cast<std::uint64_t>(cluster_weights[target]);
        if (used > capacity || own_weight > capacity - used) continue;
        const auto gain = merged_support[j] - current_support;
        if (best < 0 || gain > best_gain ||
            (gain == best_gain && target < merged_labels[best])) {
            best = j;
            best_gain = gain;
        }
    }
    if (best >= 0) {
        proposals[warp] = merged_labels[best];
        gains[warp] = best_gain;
        proposal_scores[warp] = static_cast<float>(best_gain);
    }
}

__global__ void frontier_boundary_exact_gain_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::uint64_t* edge_weights,
    const std::int32_t* labels, std::int32_t* proposals,
    float* proposal_scores, std::uint64_t* gains) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + threadIdx.x / BASC_WARP_SIZE;
    if (warp >= n) return;
    const unsigned mask = __activemask();
    const auto target = proposals[warp];
    if (target < 0) return;
    const auto current = labels[warp];
    std::uint64_t source_support = 0;
    std::uint64_t target_support = 0;
    for (auto e = offsets[warp] + lane; e < offsets[warp + 1]; e += BASC_WARP_SIZE) {
        const auto neighbor = neighbors[e];
        if (neighbor == warp) continue;
        const auto neighbor_label = labels[neighbor];
        if (neighbor_label == current) source_support += edge_weights[e];
        if (neighbor_label == target) target_support += edge_weights[e];
    }
    for (int delta = BASC_WARP_SIZE / 2; delta > 0; delta >>= 1) {
        source_support += __shfl_down_sync(mask, source_support, delta);
        target_support += __shfl_down_sync(mask, target_support, delta);
    }
    if (lane != 0) return;
    if (target_support <= source_support) {
        proposals[warp] = BASC_INVALID;
        proposal_scores[warp] = -1.0f;
        gains[warp] = 0;
        return;
    }
    const auto gain = target_support - source_support;
    gains[warp] = gain;
    proposal_scores[warp] = static_cast<float>(gain);
}

__global__ void frontier_boundary_conflict_kernel(
    std::int64_t n, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::int32_t* proposals,
    const std::uint64_t* gains, std::uint8_t* winners) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + threadIdx.x / BASC_WARP_SIZE;
    if (warp >= n) return;
    const unsigned mask = __activemask();
    if (proposals[warp] < 0) {
        if (lane == 0) winners[warp] = 0;
        return;
    }
    bool dominated = false;
    for (auto e = offsets[warp] + lane; e < offsets[warp + 1]; e += BASC_WARP_SIZE) {
        const auto u = neighbors[e];
        if (proposals[u] < 0) continue;
        if (gains[u] > gains[warp] ||
            (gains[u] == gains[warp] && u < warp)) dominated = true;
    }
    const bool any = __any_sync(mask, dominated);
    if (lane == 0) winners[warp] = any ? 0 : 1;
}

__global__ void frontier_apply_boundary_conflicts_kernel(
    std::int64_t n, const std::uint8_t* winners,
    std::int32_t* proposals) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v < n && !winners[v]) proposals[v] = BASC_INVALID;
}

__global__ void frontier_boundary_commit_kernel(
    std::int64_t count, const std::int32_t* order,
    const std::int32_t* targets, const std::uint64_t* weights,
    const std::uint64_t* prefix_weights,
    const unsigned long long* base_cluster_weights, std::uint64_t capacity,
    const std::uint64_t* gains, std::int32_t* labels,
    unsigned long long* cluster_weights, unsigned long long* moved_count,
    unsigned long long* total_gain) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto target = targets[i];
    const auto base = static_cast<std::uint64_t>(base_cluster_weights[target]);
    const auto prefix = prefix_weights[i];
    if (base > capacity || prefix > capacity - base) return;
    const auto v = order[i];
    const auto source = labels[v];
    const auto weight = static_cast<unsigned long long>(weights[i]);
    atomicAdd(&cluster_weights[source], 0ULL - weight);
    atomicAdd(&cluster_weights[target], weight);
    labels[v] = target;
    atomicAdd(moved_count, 1ULL);
    atomicAdd(total_gain, static_cast<unsigned long long>(gains[v]));
}

__global__ void frontier_mark_label_roots_kernel(
    std::int64_t n, const std::int32_t* labels, std::int32_t* root_flags) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v < n) root_flags[v] = labels[v] == v ? 1 : 0;
}

__global__ void frontier_compact_labels_kernel(
    std::int64_t n, const std::int32_t* labels,
    const std::int32_t* root_ids, std::int32_t* map) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v < n) map[v] = root_ids[labels[v]];
}

constexpr int SCLP_LOW_DEGREE = 8;
constexpr int SCLP_MEDIUM_DEGREE = 256;
constexpr std::int32_t SCLP_INVALID = 0x7fffffff;

__global__ void sclp_degree_class_kernel(
    std::int64_t n, const std::int64_t* offsets,
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

__global__ void sclp_fill_affinity_low_kernel(
    std::int64_t count, const std::int32_t* vertices,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::uint64_t* edge_weights, const std::int32_t* clusters,
    std::uint64_t* keys, std::uint64_t* values) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto v = vertices[i];
    for (auto e = offsets[v]; e < offsets[v + 1]; ++e) {
        if (neighbors[e] == v) {
            keys[e] = BASC_INVALID_KEY;
            values[e] = 0;
            continue;
        }
        keys[e] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)) << 32) |
                  static_cast<std::uint32_t>(clusters[neighbors[e]]);
        values[e] = edge_weights[e];
    }
}

__global__ void sclp_fill_affinity_medium_kernel(
    std::int64_t count, const std::int32_t* vertices,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::uint64_t* edge_weights, const std::int32_t* clusters,
    std::uint64_t* keys, std::uint64_t* values) {
    const int lane = threadIdx.x & (BASC_WARP_SIZE - 1);
    const auto warp = static_cast<std::int64_t>(blockIdx.x) *
                      BASC_WARPS_PER_BLOCK + threadIdx.x / BASC_WARP_SIZE;
    if (warp >= count) return;
    const auto v = vertices[warp];
    for (auto e = offsets[v] + lane; e < offsets[v + 1]; e += BASC_WARP_SIZE) {
        if (neighbors[e] == v) {
            keys[e] = BASC_INVALID_KEY;
            values[e] = 0;
            continue;
        }
        keys[e] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)) << 32) |
                  static_cast<std::uint32_t>(clusters[neighbors[e]]);
        values[e] = edge_weights[e];
    }
}

__global__ void sclp_fill_affinity_high_kernel(
    std::int64_t count, const std::int32_t* vertices,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::uint64_t* edge_weights, const std::int32_t* clusters,
    std::uint64_t* keys, std::uint64_t* values) {
    const auto i = static_cast<std::int64_t>(blockIdx.x);
    if (i >= count) return;
    const auto v = vertices[i];
    for (auto e = offsets[v] + threadIdx.x; e < offsets[v + 1]; e += blockDim.x) {
        if (neighbors[e] == v) {
            keys[e] = BASC_INVALID_KEY;
            values[e] = 0;
            continue;
        }
        keys[e] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(v)) << 32) |
                  static_cast<std::uint32_t>(clusters[neighbors[e]]);
        values[e] = edge_weights[e];
    }
}

__device__ __forceinline__ bool sclp_cluster_is_mover(
    std::int32_t cluster, std::uint32_t role_salt,
    std::uint32_t mover_threshold) {
    return mix32(static_cast<std::uint32_t>(cluster) ^ role_salt) < mover_threshold;
}

__global__ void sclp_affinity_baseline_kernel(
    std::int64_t count, const std::uint64_t* keys,
    const std::uint64_t* connections, const std::int32_t* clusters,
    std::uint32_t role_salt, std::uint32_t mover_threshold,
    bool filter_roles, unsigned long long* current_affinity,
    unsigned long long* best_affinity) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto v = static_cast<std::uint32_t>(keys[i] >> 32);
    const auto target = static_cast<std::int32_t>(keys[i]);
    const auto source = clusters[v];
    if (target == source) {
        current_affinity[v] = static_cast<unsigned long long>(connections[i]);
        return;
    }
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
    if (target == source || connections[i] != best_affinity[v]) return;
    if (filter_roles && sclp_cluster_is_mover(
            target, role_salt, mover_threshold)) return;
    const auto hash = mix32(v ^ mix32(static_cast<std::uint32_t>(target)) ^ tie_salt);
    const auto key = (static_cast<unsigned long long>(hash) << 32) |
                     (0xffffffffULL - static_cast<std::uint32_t>(target));
    atomicMax(best_ties + v, key);
}

__global__ void sclp_decode_gain_kernel(
    std::int64_t n, const std::int32_t* clusters,
    const unsigned long long* current_affinity,
    const unsigned long long* best_affinity,
    const unsigned long long* best_ties,
    std::uint32_t role_salt, std::uint32_t mover_threshold,
    bool filter_roles, std::int32_t* proposals,
    unsigned long long* gains, std::uint32_t* proposal_ties) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n) return;
    proposals[v] = SCLP_INVALID;
    gains[v] = 0;
    proposal_ties[v] = 0;
    const auto source = clusters[v];
    if (filter_roles && !sclp_cluster_is_mover(
            source, role_salt, mover_threshold)) return;
    if (best_affinity[v] == 0 || best_affinity[v] <= current_affinity[v]) return;
    const auto target = static_cast<std::int32_t>(
        0xffffffffULL - (best_ties[v] & 0xffffffffULL));
    if (target < 0 || target == source) return;
    proposals[v] = target;
    gains[v] = best_affinity[v] - current_affinity[v];
    proposal_ties[v] = static_cast<std::uint32_t>(best_ties[v] >> 32);
}

struct SclpProposalOrder {
    const std::int32_t* targets;
    const unsigned long long* gains;
    const std::uint32_t* ties;
    __device__ bool operator()(std::int32_t a, std::int32_t b) const {
        const auto ta = targets[a];
        const auto tb = targets[b];
        if (ta != tb) return ta < tb;
        const auto aa = gains[a];
        const auto ab = gains[b];
        if (aa != ab) return aa > ab;
        if (ties[a] != ties[b]) return ties[a] > ties[b];
        return a < b;
    }
};

struct SclpHasProposal {
    const std::int32_t* targets;
    __device__ bool operator()(std::int32_t v) const {
        return targets[v] != SCLP_INVALID;
    }
};

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
        atomicAdd(accepted, 1ULL);
        atomicAdd(predicted_gain, gains[v]);
    } else {
        atomicAdd(rejected, 1ULL);
    }
}

__global__ void sclp_singleton_favorite_kernel(
    std::int64_t n, const std::uint64_t* vertex_weights,
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
    atomicAdd(merged, 1ULL);
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

std::uint64_t host_cut(
    const WeightedGraph& graph, const std::vector<std::int32_t>& labels) {
    std::uint64_t directed = 0;
    for (std::int64_t v = 0; v < graph.vertices(); ++v) {
        for (auto e = graph.offsets[static_cast<std::size_t>(v)];
             e < graph.offsets[static_cast<std::size_t>(v + 1)]; ++e) {
            if (labels[static_cast<std::size_t>(v)] !=
                labels[static_cast<std::size_t>(graph.neighbors[static_cast<std::size_t>(e)])]) {
                directed += graph.edge_weights[static_cast<std::size_t>(e)];
            }
        }
    }
    if (directed & 1ULL) {
        throw std::runtime_error("weighted CSR cut is not symmetric");
    }
    return directed / 2;
}

std::vector<std::uint64_t> partition_loads(
    const WeightedGraph& graph, const std::vector<std::int32_t>& labels,
    int parts) {
    std::vector<std::uint64_t> loads(static_cast<std::size_t>(parts), 0);
    for (std::size_t v = 0; v < labels.size(); ++v) {
        const int p = labels[v];
        if (p < 0 || p >= parts) throw std::runtime_error("invalid weighted label");
        loads[static_cast<std::size_t>(p)] += graph.vertex_weights[v];
    }
    return loads;
}

[[maybe_unused]] std::vector<std::int32_t> gpu_weighted_refine(
    const WeightedGraph& graph, std::vector<std::int32_t> labels,
    int parts, double ratio, int rounds, int phases, int level) {
    if (parts < 2 || parts > 32) throw std::runtime_error("weighted LP supports k=2..32");
    const auto n = graph.vertices();
    const auto m = graph.edges();
    const auto total_weight = std::accumulate(
        graph.vertex_weights.begin(), graph.vertex_weights.end(), std::uint64_t{0});
    const auto capacity = static_cast<std::uint64_t>(std::floor(
        static_cast<long double>(total_weight) * ratio / parts));
    auto loads = partition_loads(graph, labels, parts);
    if (*std::max_element(loads.begin(), loads.end()) > capacity) {
        throw std::runtime_error("weighted LP received an infeasible projected partition");
    }

    std::int64_t* d_offsets = nullptr;
    std::int32_t* d_neighbors = nullptr;
    std::uint64_t* d_edge_weights = nullptr;
    std::uint64_t* d_vertex_weights = nullptr;
    std::int32_t* d_labels = nullptr;
    std::int32_t* d_targets = nullptr;
    std::int64_t* d_gains = nullptr;
    unsigned long long* d_loads = nullptr;
    unsigned long long* d_changed = nullptr;
    unsigned long long* d_cut = nullptr;
    const auto cleanup = [&]() {
        cudaFree(d_offsets); cudaFree(d_neighbors); cudaFree(d_edge_weights);
        cudaFree(d_vertex_weights); cudaFree(d_labels); cudaFree(d_targets);
        cudaFree(d_gains); cudaFree(d_loads); cudaFree(d_changed); cudaFree(d_cut);
    };
    try {
        CUDA_CHECK(cudaMalloc(&d_offsets, static_cast<std::size_t>(n + 1) * sizeof(std::int64_t)));
        CUDA_CHECK(cudaMalloc(&d_neighbors, static_cast<std::size_t>(m) * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_edge_weights, static_cast<std::size_t>(m) * sizeof(std::uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_vertex_weights, static_cast<std::size_t>(n) * sizeof(std::uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_labels, static_cast<std::size_t>(n) * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_targets, static_cast<std::size_t>(n) * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_gains, static_cast<std::size_t>(n) * sizeof(std::int64_t)));
        CUDA_CHECK(cudaMalloc(&d_loads, static_cast<std::size_t>(parts) * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_changed, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_cut, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemcpy(d_offsets, graph.offsets.data(), static_cast<std::size_t>(n + 1) * sizeof(std::int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_neighbors, graph.neighbors.data(), static_cast<std::size_t>(m) * sizeof(std::int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_edge_weights, graph.edge_weights.data(), static_cast<std::size_t>(m) * sizeof(std::uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vertex_weights, graph.vertex_weights.data(), static_cast<std::size_t>(n) * sizeof(std::uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_labels, labels.data(), static_cast<std::size_t>(n) * sizeof(std::int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_loads, loads.data(), static_cast<std::size_t>(parts) * sizeof(std::uint64_t), cudaMemcpyHostToDevice));

        const int blocks = static_cast<int>((n + 255) / 256);
        auto device_cut = [&]() {
            CUDA_CHECK(cudaMemset(d_cut, 0, sizeof(unsigned long long)));
            weighted_cut_kernel<<<blocks, 256>>>(n, d_offsets, d_neighbors,
                d_edge_weights, d_labels, d_cut);
            CUDA_CHECK(cudaGetLastError());
            unsigned long long directed = 0;
            CUDA_CHECK(cudaMemcpy(&directed, d_cut, sizeof(directed), cudaMemcpyDeviceToHost));
            if (directed & 1ULL) throw std::runtime_error("device weighted cut is not symmetric");
            return static_cast<std::uint64_t>(directed / 2);
        };

        std::uint64_t best_cut = device_cut();
        auto best_labels = labels;
        for (int round = 0; round < rounds; ++round) {
            unsigned long long round_changed = 0;
            for (int phase = 0; phase < phases; ++phase) {
                weighted_propose_kernel<<<blocks, 256>>>(
                    n, parts, phase, phases, round, d_offsets, d_neighbors,
                    d_edge_weights, d_labels, d_targets, d_gains);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(unsigned long long)));
                weighted_apply_kernel<<<blocks, 256>>>(
                    n, d_vertex_weights, d_targets, d_gains, capacity,
                    d_loads, d_labels, d_changed);
                CUDA_CHECK(cudaGetLastError());
                unsigned long long changed = 0;
                CUDA_CHECK(cudaMemcpy(&changed, d_changed, sizeof(changed), cudaMemcpyDeviceToHost));
                round_changed += changed;
            }
            const auto cut = device_cut();
            if (cut < best_cut) {
                best_cut = cut;
                CUDA_CHECK(cudaMemcpy(best_labels.data(), d_labels,
                    static_cast<std::size_t>(n) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToHost));
            }
            std::cout << "ml_refine level=" << level << " round=" << round
                      << " moved=" << round_changed << " cut=" << cut
                      << " best_cut=" << best_cut << '\n';
            if (round_changed == 0) break;
        }
        labels = std::move(best_labels);
        const auto checked = host_cut(graph, labels);
        if (checked != best_cut) throw std::runtime_error("weighted GPU/host cut mismatch");
        loads = partition_loads(graph, labels, parts);
        if (*std::max_element(loads.begin(), loads.end()) > capacity) {
            throw std::runtime_error("weighted GPU refinement violated capacity");
        }
        cleanup();
        return labels;
    } catch (...) {
        cleanup();
        throw;
    }
}

AggregateResult gpu_size_constrained_lp_aggregate(
    const WeightedGraph& graph, int rounds, std::uint64_t cluster_cap,
    std::uint32_t seed) {
    const auto n = graph.vertices();
    const auto m = graph.edges();
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("GPU LP aggregation requires at most 2^32-1 vertices");
    }
    if (n == 0) throw std::runtime_error("GPU LP aggregation received an empty graph");

    std::vector<std::int32_t> initial_clusters(static_cast<std::size_t>(n));
    std::iota(initial_clusters.begin(), initial_clusters.end(), 0);
    const auto transfer_start = std::chrono::steady_clock::now();
    thrust::device_vector<std::int64_t> offsets(graph.offsets);
    thrust::device_vector<std::int32_t> neighbors(graph.neighbors);
    thrust::device_vector<std::uint64_t> edge_weights(graph.edge_weights);
    thrust::device_vector<std::uint64_t> vertex_weights(graph.vertex_weights);
    thrust::device_vector<std::int32_t> clusters(initial_clusters);
    thrust::device_vector<std::uint64_t> cluster_weights(graph.vertex_weights);
    thrust::device_vector<std::int32_t> targets(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint64_t> best_connections(static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> best_ties(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint64_t> keys(static_cast<std::size_t>(m));
    thrust::device_vector<std::uint64_t> values(static_cast<std::size_t>(m));
    thrust::device_vector<std::uint64_t> unique_keys(static_cast<std::size_t>(m));
    thrust::device_vector<std::uint64_t> connections(static_cast<std::size_t>(m));
    thrust::device_vector<unsigned long long> changed(1);
    const auto transfer_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - transfer_start).count();
    std::cout << "ml_gpu_lp_host_device_seconds=" << transfer_seconds << '\n';

    const int vertex_blocks = static_cast<int>((n + 255) / 256);
    double kernel_seconds = 0.0;
    for (int round = 0; round < rounds; ++round) {
        const auto start = std::chrono::steady_clock::now();
        if (m > 0) {
            build_lp_edge_keys_kernel<<<vertex_blocks, 256>>>(
                n, thrust::raw_pointer_cast(offsets.data()),
                thrust::raw_pointer_cast(neighbors.data()),
                thrust::raw_pointer_cast(edge_weights.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(keys.data()),
                thrust::raw_pointer_cast(values.data()));
            CUDA_CHECK(cudaGetLastError());
            thrust::sort_by_key(thrust::device, keys.begin(), keys.end(), values.begin());
        }
        const auto reduced_end = thrust::reduce_by_key(
            thrust::device, keys.begin(), keys.end(), values.begin(),
            unique_keys.begin(), connections.begin());
        const auto unique_count = static_cast<std::int64_t>(
            reduced_end.first - unique_keys.begin());
        CUDA_CHECK(cudaMemset(
            thrust::raw_pointer_cast(best_connections.data()), 0,
            static_cast<std::size_t>(n) * sizeof(std::uint64_t)));
        CUDA_CHECK(cudaMemset(
            thrust::raw_pointer_cast(best_ties.data()), 0,
            static_cast<std::size_t>(n) * sizeof(unsigned long long)));
        if (unique_count > 0) {
            const int candidate_blocks = static_cast<int>((unique_count + 255) / 256);
            lp_best_connection_kernel<<<candidate_blocks, 256>>>(
                unique_count, cluster_cap,
                thrust::raw_pointer_cast(unique_keys.data()),
                thrust::raw_pointer_cast(connections.data()),
                thrust::raw_pointer_cast(vertex_weights.data()),
                thrust::raw_pointer_cast(cluster_weights.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(best_connections.data()));
            CUDA_CHECK(cudaGetLastError());
            lp_best_target_kernel<<<candidate_blocks, 256>>>(
                unique_count, cluster_cap, round, seed,
                thrust::raw_pointer_cast(unique_keys.data()),
                thrust::raw_pointer_cast(connections.data()),
                thrust::raw_pointer_cast(vertex_weights.data()),
                thrust::raw_pointer_cast(cluster_weights.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(best_connections.data()),
                thrust::raw_pointer_cast(best_ties.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        lp_decode_targets_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(best_connections.data()),
            thrust::raw_pointer_cast(best_ties.data()),
            thrust::raw_pointer_cast(clusters.data()),
            thrust::raw_pointer_cast(targets.data()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemset(
            thrust::raw_pointer_cast(changed.data()), 0,
            sizeof(unsigned long long)));
        lp_apply_cluster_moves_kernel<<<vertex_blocks, 256>>>(
            n, cluster_cap, thrust::raw_pointer_cast(vertex_weights.data()),
            thrust::raw_pointer_cast(cluster_weights.data()),
            thrust::raw_pointer_cast(targets.data()),
            thrust::raw_pointer_cast(clusters.data()),
            thrust::raw_pointer_cast(changed.data()));
        CUDA_CHECK(cudaGetLastError());
        unsigned long long host_changed = 0;
        CUDA_CHECK(cudaMemcpy(
            &host_changed, thrust::raw_pointer_cast(changed.data()),
            sizeof(host_changed), cudaMemcpyDeviceToHost));
        const auto round_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        kernel_seconds += round_seconds;
        std::cout << "ml_gpu_aggregate round=" << round
                  << " unique_pairs=" << unique_count
                  << " moved=" << host_changed
                  << " seconds=" << round_seconds
                  << '\n';
        if (host_changed == 0) break;
    }

    std::vector<std::int32_t> host_clusters(static_cast<std::size_t>(n));
    std::vector<std::uint64_t> host_cluster_weights(static_cast<std::size_t>(n));
    const auto copy_start = std::chrono::steady_clock::now();
    thrust::copy(clusters.begin(), clusters.end(), host_clusters.begin());
    thrust::copy(
        cluster_weights.begin(), cluster_weights.end(), host_cluster_weights.begin());
    const auto copy_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - copy_start).count();
    std::cout << "ml_gpu_lp_device_host_seconds=" << copy_seconds
              << " kernel_seconds=" << kernel_seconds << '\n';
    const auto total_before = std::accumulate(
        graph.vertex_weights.begin(), graph.vertex_weights.end(), std::uint64_t{0});
    const auto total_after = std::accumulate(
        host_cluster_weights.begin(), host_cluster_weights.end(), std::uint64_t{0});
    if (total_before != total_after) {
        throw std::runtime_error("GPU LP aggregation did not conserve vertex weight");
    }

    AggregateResult out;
    out.capacity = cluster_cap;
    std::vector<std::int32_t> new_id(static_cast<std::size_t>(n), -1);
    for (std::int64_t c = 0; c < n; ++c) {
        const auto weight = host_cluster_weights[static_cast<std::size_t>(c)];
        if (weight == 0) continue;
        if (weight > cluster_cap) {
            throw std::runtime_error("GPU LP aggregation exceeded cluster capacity");
        }
        new_id[static_cast<std::size_t>(c)] = out.coarse_vertices++;
        out.maximum_weight = std::max(out.maximum_weight, weight);
    }
    out.map.resize(static_cast<std::size_t>(n));
    for (std::int64_t v = 0; v < n; ++v) {
        const auto cluster = host_clusters[static_cast<std::size_t>(v)];
        if (cluster < 0 || cluster >= n) {
            throw std::runtime_error("GPU LP aggregation produced an invalid cluster");
        }
        const auto coarse = new_id[static_cast<std::size_t>(cluster)];
        if (coarse < 0) {
            throw std::runtime_error("GPU LP aggregation targeted an empty cluster");
        }
        out.map[static_cast<std::size_t>(v)] = coarse;
    }
    return out;
}

AggregateResult gpu_basc_aggregate(
    const WeightedGraph& graph, int parts, int candidate_count,
    std::uint32_t seed, int level, BascStats& stats) {
    if (candidate_count != 1 && candidate_count != 2 && candidate_count != 4) {
        throw std::runtime_error("BASC candidate count must be 1, 2, or 4");
    }
    const auto n = graph.vertices();
    const auto m = graph.edges();
    if (n <= 0 || n > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("BASC received an unsupported graph size");
    }
    const int blocks = static_cast<int>((n + 255) / 256);
    std::vector<std::int64_t> degrees(static_cast<std::size_t>(n));
    for (std::int64_t v = 0; v < n; ++v) {
        degrees[static_cast<std::size_t>(v)] =
            graph.offsets[static_cast<std::size_t>(v + 1)] -
            graph.offsets[static_cast<std::size_t>(v)];
    }

    const auto transfer_start = std::chrono::steady_clock::now();
    thrust::device_vector<std::int64_t> offsets(graph.offsets);
    thrust::device_vector<std::int32_t> neighbors(graph.neighbors);
    thrust::device_vector<std::uint64_t> edge_weights(graph.edge_weights);
    thrust::device_vector<std::uint64_t> vertex_weights(graph.vertex_weights);
    thrust::device_vector<std::int64_t> device_degrees(degrees);
    thrust::device_vector<float> priorities(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint8_t> anchors(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> candidates(
        static_cast<std::size_t>(n) * BASC_MAX_K);
    thrust::device_vector<float> affinities(
        static_cast<std::size_t>(n) * BASC_MAX_K);
    thrust::device_vector<std::uint8_t> candidate_counts(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> proposals(static_cast<std::size_t>(n));
    thrust::device_vector<float> confidence(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint8_t> buckets(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> aggregates(static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> aggregate_weights(
        static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> counters(5);
    const auto transfer_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - transfer_start).count();
    std::cout << "ml_gpu_basc_host_device_seconds level=" << level
              << " seconds=" << transfer_seconds << '\n';

    const auto anchor_start = std::chrono::steady_clock::now();
    basc_priority_kernel<<<blocks, 256>>>(
        n, thrust::raw_pointer_cast(offsets.data()), seed, 0.25f,
        thrust::raw_pointer_cast(priorities.data()));
    CUDA_CHECK(cudaGetLastError());
    basc_anchor_election_kernel<<<blocks, 256>>>(
        n, thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(neighbors.data()),
        thrust::raw_pointer_cast(priorities.data()),
        thrust::raw_pointer_cast(anchors.data()));
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::uint8_t> host_anchors(static_cast<std::size_t>(n));
    thrust::copy(anchors.begin(), anchors.end(), host_anchors.begin());
    stats.anchors = static_cast<std::uint64_t>(std::count(
        host_anchors.begin(), host_anchors.end(), static_cast<std::uint8_t>(1)));
    stats.anchor_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - anchor_start).count();

    const auto total_weight = std::accumulate(
        graph.vertex_weights.begin(), graph.vertex_weights.end(), std::uint64_t{0});
    const auto maximum_vertex = *std::max_element(
        graph.vertex_weights.begin(), graph.vertex_weights.end());
    if (stats.anchors == 0) throw std::runtime_error("BASC elected no anchors");
    const auto anchor_bound = static_cast<std::uint64_t>(std::ceil(
        2.0 * static_cast<long double>(total_weight) / stats.anchors));
    const auto partition_bound = std::max<std::uint64_t>(
        1, total_weight / static_cast<std::uint64_t>(20 * parts));
    stats.cluster_cap = std::max(maximum_vertex,
        std::min(anchor_bound, partition_bound));

    const auto candidate_start = std::chrono::steady_clock::now();
    basc_anchor_candidates_kernel<<<blocks, 256>>>(
        n, thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(neighbors.data()),
        thrust::raw_pointer_cast(edge_weights.data()),
        thrust::raw_pointer_cast(device_degrees.data()),
        thrust::raw_pointer_cast(anchors.data()), candidate_count,
        thrust::raw_pointer_cast(candidates.data()),
        thrust::raw_pointer_cast(affinities.data()),
        thrust::raw_pointer_cast(candidate_counts.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.candidate_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - candidate_start).count();

    const auto support_start = std::chrono::steady_clock::now();
    basc_support_kernel<<<blocks, 256>>>(
        n, thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(neighbors.data()),
        thrust::raw_pointer_cast(edge_weights.data()),
        thrust::raw_pointer_cast(anchors.data()), candidate_count, 0.35f,
        thrust::raw_pointer_cast(candidates.data()),
        thrust::raw_pointer_cast(affinities.data()),
        thrust::raw_pointer_cast(proposals.data()),
        thrust::raw_pointer_cast(confidence.data()),
        thrust::raw_pointer_cast(buckets.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.support_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - support_start).count();

    basc_initialize_aggregates_kernel<<<blocks, 256>>>(
        n, thrust::raw_pointer_cast(vertex_weights.data()),
        thrust::raw_pointer_cast(anchors.data()),
        thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(aggregate_weights.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counters.data()), 0,
                          5 * sizeof(unsigned long long)));
    std::vector<std::uint8_t> host_candidate_counts(static_cast<std::size_t>(n));
    std::vector<std::int32_t> host_proposals(static_cast<std::size_t>(n));
    std::vector<float> host_confidence(static_cast<std::size_t>(n));
    thrust::copy(candidate_counts.begin(), candidate_counts.end(),
                 host_candidate_counts.begin());
    thrust::copy(proposals.begin(), proposals.end(), host_proposals.begin());
    thrust::copy(confidence.begin(), confidence.end(), host_confidence.begin());
    std::vector<double> valid_confidences;
    valid_confidences.reserve(static_cast<std::size_t>(n));
    for (std::int64_t v = 0; v < n; ++v) {
        const auto count = std::min<int>(host_candidate_counts[static_cast<std::size_t>(v)], 3);
        ++stats.candidate_histogram[count];
        if (host_proposals[static_cast<std::size_t>(v)] >= 0 && !host_anchors[static_cast<std::size_t>(v)]) {
            ++stats.first_round_eligible;
        }
        if (host_proposals[static_cast<std::size_t>(v)] >= 0) {
            valid_confidences.push_back(host_confidence[static_cast<std::size_t>(v)]);
        }
    }
    if (!valid_confidences.empty()) {
        const auto quantile = [&valid_confidences](double q) {
            const auto index = static_cast<std::size_t>(q * (valid_confidences.size() - 1));
            std::nth_element(valid_confidences.begin(),
                             valid_confidences.begin() + index,
                             valid_confidences.end());
            return static_cast<double>(valid_confidences[index]);
        };
        stats.confidence_p50 = quantile(0.50);
        stats.confidence_p90 = quantile(0.90);
    }
    const auto admission_start = std::chrono::steady_clock::now();
    for (int bucket = 3; bucket >= 0; --bucket) {
        basc_admit_bucket_kernel<<<blocks, 256>>>(
            n, bucket, stats.cluster_cap,
            thrust::raw_pointer_cast(vertex_weights.data()),
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(buckets.data()),
            thrust::raw_pointer_cast(aggregates.data()),
            thrust::raw_pointer_cast(aggregate_weights.data()),
            thrust::raw_pointer_cast(counters.data()) + 0,
            thrust::raw_pointer_cast(counters.data()) + 1);
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long accepted_first_round = 0;
    unsigned long long rejected_first_round = 0;
    CUDA_CHECK(cudaMemcpy(&accepted_first_round,
        thrust::raw_pointer_cast(counters.data()) + 0, sizeof(accepted_first_round),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&rejected_first_round,
        thrust::raw_pointer_cast(counters.data()) + 1, sizeof(rejected_first_round),
        cudaMemcpyDeviceToHost));
    stats.accepted_first_round = accepted_first_round;
    stats.capacity_reject_first_round = rejected_first_round;
    stats.admission_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - admission_start).count();

    CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counters.data()) + 2, 0,
                          2 * sizeof(unsigned long long)));
    const auto expansion_start = std::chrono::steady_clock::now();
    basc_expansion_kernel<<<blocks, 256>>>(
        n, 0.25f, stats.cluster_cap,
        thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(neighbors.data()),
        thrust::raw_pointer_cast(edge_weights.data()),
        thrust::raw_pointer_cast(vertex_weights.data()),
        thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(aggregate_weights.data()),
        thrust::raw_pointer_cast(counters.data()) + 2,
        thrust::raw_pointer_cast(counters.data()) + 3,
        thrust::raw_pointer_cast(counters.data()) + 4);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long accepted_expansion = 0;
    unsigned long long rejected_expansion = 0;
    unsigned long long eligible_expansion = 0;
    CUDA_CHECK(cudaMemcpy(&accepted_expansion,
        thrust::raw_pointer_cast(counters.data()) + 2, sizeof(accepted_expansion),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&rejected_expansion,
        thrust::raw_pointer_cast(counters.data()) + 3, sizeof(rejected_expansion),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&eligible_expansion,
        thrust::raw_pointer_cast(counters.data()) + 4, sizeof(eligible_expansion),
        cudaMemcpyDeviceToHost));
    stats.accepted_expansion = accepted_expansion;
    stats.capacity_reject_expansion = rejected_expansion;
    stats.expansion_eligible = eligible_expansion;
    stats.expansion_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - expansion_start).count();

    CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counters.data()), 0,
                          sizeof(unsigned long long)));
    basc_singleton_kernel<<<blocks, 256>>>(
        n, thrust::raw_pointer_cast(vertex_weights.data()),
        thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(aggregate_weights.data()),
        thrust::raw_pointer_cast(counters.data()));
    CUDA_CHECK(cudaGetLastError());
    unsigned long long singleton_count = 0;
    CUDA_CHECK(cudaMemcpy(&singleton_count,
        thrust::raw_pointer_cast(counters.data()), sizeof(singleton_count),
        cudaMemcpyDeviceToHost));
    stats.singleton_count = singleton_count;

    std::vector<std::int32_t> host_aggregates(static_cast<std::size_t>(n));
    std::vector<unsigned long long> host_aggregate_weights(
        static_cast<std::size_t>(n));
    thrust::copy(aggregates.begin(), aggregates.end(), host_aggregates.begin());
    thrust::copy(aggregate_weights.begin(), aggregate_weights.end(),
                 host_aggregate_weights.begin());
    const auto total_after = std::accumulate(
        host_aggregate_weights.begin(), host_aggregate_weights.end(),
        std::uint64_t{0});
    if (total_after != total_weight) {
        throw std::runtime_error("BASC did not conserve aggregate weight");
    }

    AggregateResult out;
    out.capacity = stats.cluster_cap;
    std::vector<std::int32_t> new_id(static_cast<std::size_t>(n), BASC_INVALID);
    for (std::int64_t v = 0; v < n; ++v) {
        const auto root = host_aggregates[static_cast<std::size_t>(v)];
        if (root < 0 || root >= n || host_aggregates[static_cast<std::size_t>(root)] != root) {
            throw std::runtime_error("BASC produced a non-root aggregate id");
        }
        if (root == v) {
            const auto weight = host_aggregate_weights[static_cast<std::size_t>(v)];
            if (weight == 0 || weight > stats.cluster_cap) {
                throw std::runtime_error("BASC aggregate violates its capacity");
            }
            new_id[static_cast<std::size_t>(v)] = out.coarse_vertices++;
            out.maximum_weight = std::max(
                out.maximum_weight, static_cast<std::uint64_t>(weight));
        }
    }
    out.map.resize(static_cast<std::size_t>(n));
    for (std::int64_t v = 0; v < n; ++v) {
        const auto root = host_aggregates[static_cast<std::size_t>(v)];
        const auto compact = new_id[static_cast<std::size_t>(root)];
        if (compact < 0) throw std::runtime_error("BASC root was not compacted");
        out.map[static_cast<std::size_t>(v)] = compact;
    }
    std::cout << "ml_gpu_basc level=" << level
              << " k=" << candidate_count
              << " anchors=" << stats.anchors
              << " candidate_0=" << stats.candidate_histogram[0]
              << " candidate_1=" << stats.candidate_histogram[1]
              << " candidate_2=" << stats.candidate_histogram[2]
              << " candidate_3plus=" << stats.candidate_histogram[3]
              << " first_round_eligible=" << stats.first_round_eligible
              << " accepted_first_round=" << stats.accepted_first_round
              << " capacity_reject_first_round=" << stats.capacity_reject_first_round
              << " expansion_eligible=" << stats.expansion_eligible
              << " accepted_expansion=" << stats.accepted_expansion
              << " capacity_reject_expansion=" << stats.capacity_reject_expansion
              << " singletons=" << stats.singleton_count
              << " confidence_p50=" << stats.confidence_p50
              << " confidence_p90=" << stats.confidence_p90
              << " cluster_cap=" << stats.cluster_cap
              << " anchor_seconds=" << stats.anchor_seconds
              << " candidate_seconds=" << stats.candidate_seconds
              << " support_seconds=" << stats.support_seconds
              << " admission_seconds=" << stats.admission_seconds
              << " expansion_seconds=" << stats.expansion_seconds << '\n';
    return out;
}

DeviceWeightedGraph make_device_weighted(const WeightedGraph& graph) {
    DeviceWeightedGraph out;
    out.offsets = graph.offsets;
    out.neighbors = graph.neighbors;
    out.edge_weights = graph.edge_weights;
    out.vertex_weights = graph.vertex_weights;
    return out;
}

WeightedGraph copy_device_weighted(const DeviceWeightedGraph& graph) {
    WeightedGraph out;
    out.offsets.resize(graph.offsets.size());
    out.neighbors.resize(graph.neighbors.size());
    out.edge_weights.resize(graph.edge_weights.size());
    out.vertex_weights.resize(graph.vertex_weights.size());
    thrust::copy(graph.offsets.begin(), graph.offsets.end(), out.offsets.begin());
    thrust::copy(graph.neighbors.begin(), graph.neighbors.end(), out.neighbors.begin());
    thrust::copy(graph.edge_weights.begin(), graph.edge_weights.end(), out.edge_weights.begin());
    thrust::copy(graph.vertex_weights.begin(), graph.vertex_weights.end(), out.vertex_weights.begin());
    return out;
}

void collect_basc_device_diagnostics(
    const thrust::device_vector<std::uint8_t>& anchors,
    const thrust::device_vector<std::uint8_t>& candidate_counts,
    const thrust::device_vector<std::int32_t>& proposals,
    const thrust::device_vector<float>& confidence,
    std::int64_t n, BascStats& stats) {
    std::vector<std::uint8_t> host_anchors(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> host_counts(static_cast<std::size_t>(n));
    std::vector<std::int32_t> host_proposals(static_cast<std::size_t>(n));
    std::vector<float> host_confidence(static_cast<std::size_t>(n));
    thrust::copy(anchors.begin(), anchors.end(), host_anchors.begin());
    thrust::copy(candidate_counts.begin(), candidate_counts.end(), host_counts.begin());
    thrust::copy(proposals.begin(), proposals.end(), host_proposals.begin());
    thrust::copy(confidence.begin(), confidence.end(), host_confidence.begin());
    std::vector<double> valid_confidences;
    valid_confidences.reserve(static_cast<std::size_t>(n));
    for (std::int64_t v = 0; v < n; ++v) {
        const auto index = static_cast<std::size_t>(v);
        const auto count = std::min<int>(host_counts[index], 3);
        ++stats.candidate_histogram[count];
        if (host_proposals[index] >= 0 && !host_anchors[index]) {
            ++stats.first_round_eligible;
        }
        if (host_proposals[index] >= 0) {
            valid_confidences.push_back(host_confidence[index]);
        }
    }
    if (!valid_confidences.empty()) {
        const auto quantile = [&valid_confidences](double q) {
            const auto index = static_cast<std::size_t>(
                q * static_cast<double>(valid_confidences.size() - 1));
            std::nth_element(valid_confidences.begin(),
                             valid_confidences.begin() + index,
                             valid_confidences.end());
            return static_cast<double>(valid_confidences[index]);
        };
        stats.confidence_p50 = quantile(0.50);
        stats.confidence_p90 = quantile(0.90);
    }
}

DeviceAggregateResult gpu_basc_aggregate_device(
    const DeviceWeightedGraph& graph, int parts, int candidate_count,
    std::uint32_t seed, int level, BascStats& stats, bool diagnostics) {
    if (candidate_count != 1 && candidate_count != 2 && candidate_count != 4) {
        throw std::runtime_error("BASC GPU candidate count must be 1, 2, or 4");
    }
    const auto n = graph.vertices();
    if (n <= 0 || n > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("BASC GPU received an unsupported graph size");
    }
    const int vertex_blocks = static_cast<int>((n + 255) / 256);
    const int warp_blocks = static_cast<int>(
        (n + BASC_WARPS_PER_BLOCK - 1) / BASC_WARPS_PER_BLOCK);
    const auto total_weight = thrust::reduce(
        graph.vertex_weights.begin(), graph.vertex_weights.end(),
        std::uint64_t{0}, thrust::plus<std::uint64_t>());
    const auto maximum_vertex = thrust::reduce(
        graph.vertex_weights.begin(), graph.vertex_weights.end(),
        std::uint64_t{0}, thrust::maximum<std::uint64_t>());

    thrust::device_vector<float> priorities(static_cast<std::size_t>(n));
    thrust::device_vector<float> inverse_degree(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint8_t> anchors(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> candidates(
        static_cast<std::size_t>(n) * BASC_MAX_K);
    thrust::device_vector<float> affinities(
        static_cast<std::size_t>(n) * BASC_MAX_K);
    thrust::device_vector<std::uint8_t> candidate_counts(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> proposals(static_cast<std::size_t>(n));
    thrust::device_vector<float> confidence(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint8_t> buckets(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> aggregates(static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> aggregate_weights(
        static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> expansion_targets(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint8_t> expansion_eligible(static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> counters(5);
    const auto timer_start = std::chrono::steady_clock::now();

    basc_priority_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.offsets.data()), seed, 0.25f,
        thrust::raw_pointer_cast(priorities.data()));
    CUDA_CHECK(cudaGetLastError());
    basc_warp_anchor_election_kernel<<<warp_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.offsets.data()),
        thrust::raw_pointer_cast(graph.neighbors.data()),
        thrust::raw_pointer_cast(priorities.data()),
        thrust::raw_pointer_cast(anchors.data()));
    CUDA_CHECK(cudaGetLastError());
    stats.anchors = static_cast<std::uint64_t>(thrust::count(
        anchors.begin(), anchors.end(), static_cast<std::uint8_t>(1)));
    if (stats.anchors == 0) throw std::runtime_error("BASC GPU elected no anchors");

    const auto anchor_bound = static_cast<std::uint64_t>(std::ceil(
        2.0 * static_cast<long double>(total_weight) / stats.anchors));
    const auto partition_bound = std::max<std::uint64_t>(
        1, total_weight / static_cast<std::uint64_t>(20 * parts));
    stats.cluster_cap = std::max(maximum_vertex,
        std::min(anchor_bound, partition_bound));

    basc_inverse_sqrt_degree_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.offsets.data()),
        thrust::raw_pointer_cast(inverse_degree.data()));
    CUDA_CHECK(cudaGetLastError());
    basc_warp_anchor_candidates_kernel<<<warp_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.offsets.data()),
        thrust::raw_pointer_cast(graph.neighbors.data()),
        thrust::raw_pointer_cast(graph.edge_weights.data()),
        thrust::raw_pointer_cast(inverse_degree.data()),
        thrust::raw_pointer_cast(anchors.data()), candidate_count,
        thrust::raw_pointer_cast(candidates.data()),
        thrust::raw_pointer_cast(affinities.data()),
        thrust::raw_pointer_cast(candidate_counts.data()));
    CUDA_CHECK(cudaGetLastError());
    basc_warp_support_kernel<<<warp_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.offsets.data()),
        thrust::raw_pointer_cast(graph.neighbors.data()),
        thrust::raw_pointer_cast(graph.edge_weights.data()),
        thrust::raw_pointer_cast(anchors.data()), candidate_count, 0.35f,
        thrust::raw_pointer_cast(candidates.data()),
        thrust::raw_pointer_cast(affinities.data()),
        thrust::raw_pointer_cast(proposals.data()),
        thrust::raw_pointer_cast(confidence.data()),
        thrust::raw_pointer_cast(buckets.data()));
    CUDA_CHECK(cudaGetLastError());
    if (diagnostics) {
        collect_basc_device_diagnostics(
            anchors, candidate_counts, proposals, confidence, n, stats);
    }

    basc_initialize_aggregates_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.vertex_weights.data()),
        thrust::raw_pointer_cast(anchors.data()),
        thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(aggregate_weights.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counters.data()), 0,
                          5 * sizeof(unsigned long long)));
    for (int bucket = 3; bucket >= 0; --bucket) {
        basc_admit_bucket_kernel<<<vertex_blocks, 256>>>(
            n, bucket, stats.cluster_cap,
            thrust::raw_pointer_cast(graph.vertex_weights.data()),
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(buckets.data()),
            thrust::raw_pointer_cast(aggregates.data()),
            thrust::raw_pointer_cast(aggregate_weights.data()),
            thrust::raw_pointer_cast(counters.data()) + 0,
            thrust::raw_pointer_cast(counters.data()) + 1);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(&stats.accepted_first_round,
        thrust::raw_pointer_cast(counters.data()) + 0,
        sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stats.capacity_reject_first_round,
        thrust::raw_pointer_cast(counters.data()) + 1,
        sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    basc_warp_expansion_propose_kernel<<<warp_blocks, 256>>>(
        n, 0.25f, thrust::raw_pointer_cast(graph.offsets.data()),
        thrust::raw_pointer_cast(graph.neighbors.data()),
        thrust::raw_pointer_cast(graph.edge_weights.data()),
        thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(expansion_targets.data()),
        thrust::raw_pointer_cast(expansion_eligible.data()),
        thrust::raw_pointer_cast(counters.data()) + 4);
    CUDA_CHECK(cudaGetLastError());
    basc_expansion_commit_kernel<<<vertex_blocks, 256>>>(
        n, stats.cluster_cap,
        thrust::raw_pointer_cast(graph.vertex_weights.data()),
        thrust::raw_pointer_cast(expansion_targets.data()),
        thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(aggregate_weights.data()),
        thrust::raw_pointer_cast(counters.data()) + 2,
        thrust::raw_pointer_cast(counters.data()) + 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&stats.expansion_eligible,
        thrust::raw_pointer_cast(counters.data()) + 4,
        sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stats.accepted_expansion,
        thrust::raw_pointer_cast(counters.data()) + 2,
        sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&stats.capacity_reject_expansion,
        thrust::raw_pointer_cast(counters.data()) + 3,
        sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counters.data()), 0,
                          sizeof(unsigned long long)));
    basc_singleton_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.vertex_weights.data()),
        thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(aggregate_weights.data()),
        thrust::raw_pointer_cast(counters.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&stats.singleton_count,
        thrust::raw_pointer_cast(counters.data()), sizeof(unsigned long long),
        cudaMemcpyDeviceToHost));

    thrust::device_vector<std::int32_t> root_flags(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> root_ids(static_cast<std::size_t>(n));
    DeviceAggregateResult out;
    out.map.resize(static_cast<std::size_t>(n));
    basc_mark_roots_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(root_flags.data()));
    CUDA_CHECK(cudaGetLastError());
    thrust::exclusive_scan(root_flags.begin(), root_flags.end(), root_ids.begin());
    std::int32_t last_root_id = 0;
    std::int32_t last_root_flag = 0;
    CUDA_CHECK(cudaMemcpy(&last_root_id,
        thrust::raw_pointer_cast(root_ids.data()) + (n - 1), sizeof(last_root_id),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_root_flag,
        thrust::raw_pointer_cast(root_flags.data()) + (n - 1), sizeof(last_root_flag),
        cudaMemcpyDeviceToHost));
    out.coarse_vertices = last_root_id + last_root_flag;
    if (out.coarse_vertices <= 0) {
        throw std::runtime_error("BASC GPU produced no coarse vertices");
    }
    basc_compact_map_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(root_ids.data()),
        thrust::raw_pointer_cast(out.map.data()));
    CUDA_CHECK(cudaGetLastError());
    out.vertex_weights.resize(static_cast<std::size_t>(out.coarse_vertices));
    thrust::fill(out.vertex_weights.begin(), out.vertex_weights.end(), std::uint64_t{0});
    basc_coarse_vertex_weights_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.vertex_weights.data()),
        thrust::raw_pointer_cast(out.map.data()),
        thrust::raw_pointer_cast(out.vertex_weights.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    out.maximum_weight = thrust::reduce(
        out.vertex_weights.begin(), out.vertex_weights.end(),
        std::uint64_t{0}, thrust::maximum<std::uint64_t>());
    out.capacity = stats.cluster_cap;
    const auto aggregate_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - timer_start).count();
    std::cout << "ml_gpu_basc_device level=" << level
              << " k=" << candidate_count
              << " anchors=" << stats.anchors
              << " coarse_vertices=" << out.coarse_vertices
              << " cluster_cap=" << stats.cluster_cap
              << " accepted_first_round=" << stats.accepted_first_round
              << " capacity_reject_first_round=" << stats.capacity_reject_first_round
              << " accepted_expansion=" << stats.accepted_expansion
              << " capacity_reject_expansion=" << stats.capacity_reject_expansion
              << " singletons=" << stats.singleton_count
              << " aggregate_seconds=" << aggregate_seconds;
    if (diagnostics) {
        std::cout << " candidate_0=" << stats.candidate_histogram[0]
                  << " candidate_1=" << stats.candidate_histogram[1]
                  << " candidate_2=" << stats.candidate_histogram[2]
                  << " candidate_3plus=" << stats.candidate_histogram[3]
                  << " first_round_eligible=" << stats.first_round_eligible
                  << " expansion_eligible=" << stats.expansion_eligible
                  << " confidence_p50=" << stats.confidence_p50
                  << " confidence_p90=" << stats.confidence_p90;
    }
    std::cout << '\n';
    return out;
}

DeviceAggregateResult gpu_frontier_aggregate_device(
    const DeviceWeightedGraph& graph, int parts, std::uint32_t seed,
    int level, FrontierStats& stats, bool diagnostics) {
    const auto n = graph.vertices();
    if (n <= 0 || n > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("frontier coarsening received an unsupported graph size");
    }
    const auto read_environment_double = [](
        const char* name, double fallback, double minimum, double maximum) {
        const char* raw = std::getenv(name);
        if (raw == nullptr) return fallback;
        std::size_t consumed = 0;
        const double value = std::stod(raw, &consumed);
        if (raw[consumed] != '\0' || value < minimum || value > maximum) {
            throw std::runtime_error(std::string("invalid ") + name);
        }
        return value;
    };
    const double contraction_factor = read_environment_double(
        "FRONTIER_CONTRACTION_FACTOR", 8.0, 2.0, 64.0);
    const double capacity_slack = read_environment_double(
        "FRONTIER_CAPACITY_SLACK", 0.20, 0.0, 4.0);
    const double hot_ratio = read_environment_double(
        "FRONTIER_HOT_RATIO", 0.50, 0.0, 1.0);
    constexpr float confidence_threshold = 0.60f;
    constexpr std::uint8_t max_postpone = 2;
    const auto cutoff = std::max<std::int64_t>(32, parts * 8);
    stats.target_clusters = std::min<std::int64_t>(
        n, std::max<std::int64_t>(
            cutoff, static_cast<std::int64_t>(std::ceil(n / contraction_factor))));
    stats.hot_seeds = static_cast<std::uint64_t>(std::llround(
        hot_ratio * static_cast<double>(stats.target_clusters)));
    stats.hot_seeds = std::min<std::uint64_t>(
        stats.hot_seeds, static_cast<std::uint64_t>(stats.target_clusters));
    stats.cold_seeds = static_cast<std::uint64_t>(
        stats.target_clusters - static_cast<std::int64_t>(stats.hot_seeds));

    const int vertex_blocks = static_cast<int>((n + 255) / 256);
    const int warp_blocks = static_cast<int>(
        (n + BASC_WARPS_PER_BLOCK - 1) / BASC_WARPS_PER_BLOCK);
    const auto total_weight = thrust::reduce(
        graph.vertex_weights.begin(), graph.vertex_weights.end(),
        std::uint64_t{0}, thrust::plus<std::uint64_t>());
    const auto maximum_vertex = thrust::reduce(
        graph.vertex_weights.begin(), graph.vertex_weights.end(),
        std::uint64_t{0}, thrust::maximum<std::uint64_t>());
    const auto average_capacity = static_cast<std::uint64_t>(std::ceil(
        (1.0 + capacity_slack) * static_cast<long double>(total_weight) /
        static_cast<long double>(stats.target_clusters)));
    stats.cluster_cap = std::max(maximum_vertex, average_capacity);

    thrust::device_vector<std::int64_t> degrees(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint8_t> local_maximum(static_cast<std::size_t>(n));
    thrust::device_vector<float> ranking_scores(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> ranked_vertices(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> seed_vertices(
        static_cast<std::size_t>(stats.target_clusters));
    thrust::device_vector<std::uint8_t> covered(static_cast<std::size_t>(n), 0);
    thrust::device_vector<std::uint8_t> cold_local_maximum(
        static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> labels(
        static_cast<std::size_t>(n), BASC_INVALID);
    thrust::device_vector<std::uint8_t> seed_mask(static_cast<std::size_t>(n), 0);
    thrust::device_vector<unsigned long long> cluster_weights(
        static_cast<std::size_t>(n), 0ULL);
    thrust::device_vector<float> emergency_priority(static_cast<std::size_t>(n));

    const auto seed_start = std::chrono::steady_clock::now();
    frontier_degree_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.offsets.data()),
        thrust::raw_pointer_cast(degrees.data()));
    CUDA_CHECK(cudaGetLastError());
    frontier_local_degree_max_kernel<<<warp_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.offsets.data()),
        thrust::raw_pointer_cast(graph.neighbors.data()),
        thrust::raw_pointer_cast(degrees.data()),
        thrust::raw_pointer_cast(local_maximum.data()));
    CUDA_CHECK(cudaGetLastError());
    const auto maximum_degree = thrust::reduce(
        degrees.begin(), degrees.end(), std::int64_t{0},
        thrust::maximum<std::int64_t>());
    const double inverse_log_max_degree = maximum_degree > 0
        ? 1.0 / std::log1p(static_cast<double>(maximum_degree)) : 0.0;
    frontier_hot_scores_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(degrees.data()),
        thrust::raw_pointer_cast(local_maximum.data()), inverse_log_max_degree,
        thrust::raw_pointer_cast(ranking_scores.data()));
    CUDA_CHECK(cudaGetLastError());
    thrust::sequence(ranked_vertices.begin(), ranked_vertices.end());
    thrust::stable_sort_by_key(
        thrust::device, ranking_scores.begin(), ranking_scores.end(),
        ranked_vertices.begin(), thrust::greater<float>());
    thrust::copy_n(
        ranked_vertices.begin(), static_cast<std::ptrdiff_t>(stats.hot_seeds),
        seed_vertices.begin());
    if (stats.hot_seeds > 0) {
        const int blocks = static_cast<int>((stats.hot_seeds + 255) / 256);
        frontier_initialize_seeds_kernel<<<blocks, 256>>>(
            stats.hot_seeds, thrust::raw_pointer_cast(seed_vertices.data()),
            thrust::raw_pointer_cast(graph.vertex_weights.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(seed_mask.data()),
            thrust::raw_pointer_cast(cluster_weights.data()));
        CUDA_CHECK(cudaGetLastError());
        const int hot_warp_blocks = static_cast<int>(
            (stats.hot_seeds + BASC_WARPS_PER_BLOCK - 1) /
            BASC_WARPS_PER_BLOCK);
        frontier_cover_hot_kernel<<<hot_warp_blocks, 256>>>(
            stats.hot_seeds, thrust::raw_pointer_cast(seed_vertices.data()),
            thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(covered.data()));
        CUDA_CHECK(cudaGetLastError());
    }
    std::uint64_t cold_selected = 0;
    while (cold_selected < stats.cold_seeds) {
        ++stats.cold_seed_rounds;
        frontier_cold_scores_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(degrees.data()),
            thrust::raw_pointer_cast(covered.data()),
            thrust::raw_pointer_cast(labels.data()), inverse_log_max_degree,
            seed + static_cast<std::uint32_t>(stats.cold_seed_rounds),
            thrust::raw_pointer_cast(ranking_scores.data()));
        CUDA_CHECK(cudaGetLastError());
        frontier_cold_local_max_kernel<<<warp_blocks, 256>>>(
            n, thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(covered.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(ranking_scores.data()),
            thrust::raw_pointer_cast(cold_local_maximum.data()));
        CUDA_CHECK(cudaGetLastError());
        const auto local_count = static_cast<std::uint64_t>(thrust::count(
            cold_local_maximum.begin(), cold_local_maximum.end(),
            static_cast<std::uint8_t>(1)));
        frontier_cold_rank_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(degrees.data()),
            thrust::raw_pointer_cast(covered.data()),
            thrust::raw_pointer_cast(cold_local_maximum.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(ranking_scores.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::sequence(ranked_vertices.begin(), ranked_vertices.end());
        thrust::stable_sort_by_key(
            thrust::device, ranking_scores.begin(), ranking_scores.end(),
            ranked_vertices.begin(), thrust::greater<float>());
        const auto remaining = stats.cold_seeds - cold_selected;
        // Take a spatially independent local-max batch.  If the uncovered
        // induced graph has no candidate, use the ranking as a deterministic
        // fallback so the requested seed count is still reached.
        const auto take = static_cast<std::uint64_t>(
            std::min<std::uint64_t>(remaining, local_count > 0 ? local_count : remaining));
        thrust::copy_n(
            ranked_vertices.begin(), static_cast<std::ptrdiff_t>(take),
            seed_vertices.begin() + static_cast<std::ptrdiff_t>(
                stats.hot_seeds + cold_selected));
        const int blocks = static_cast<int>((take + 255) / 256);
        auto* selected = thrust::raw_pointer_cast(seed_vertices.data()) +
                         stats.hot_seeds + cold_selected;
        frontier_initialize_seeds_kernel<<<blocks, 256>>>(
            take, selected, thrust::raw_pointer_cast(graph.vertex_weights.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(seed_mask.data()),
            thrust::raw_pointer_cast(cluster_weights.data()));
        CUDA_CHECK(cudaGetLastError());
        const int cover_blocks = static_cast<int>(
            (take + BASC_WARPS_PER_BLOCK - 1) / BASC_WARPS_PER_BLOCK);
        frontier_cover_hot_kernel<<<cover_blocks, 256>>>(
            take, selected, thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(covered.data()));
        CUDA_CHECK(cudaGetLastError());
        cold_selected += take;
        if (stats.cold_seed_rounds > 32) {
            throw std::runtime_error("frontier cold seed selection exceeded 32 rounds");
        }
    }
    basc_priority_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.offsets.data()), seed, 0.25f,
        thrust::raw_pointer_cast(emergency_priority.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.seed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - seed_start).count();

    thrust::device_vector<std::int32_t> frontier(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> active(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> next_active(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> accepted_frontier(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> active_epoch(
        static_cast<std::size_t>(n), -1);
    thrust::device_vector<std::uint8_t> postpone(static_cast<std::size_t>(n), 0);
    thrust::device_vector<std::int32_t> proposals(
        static_cast<std::size_t>(n), BASC_INVALID);
    thrust::device_vector<float> proposal_scores(static_cast<std::size_t>(n), -1.0f);
    thrust::device_vector<std::int32_t> order(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> ordered_targets(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint64_t> ordered_weights(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint64_t> prefix_weights(static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> base_cluster_weights(
        static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> proposal_counters(2, 0ULL);
    thrust::device_vector<unsigned long long> commit_counters(2, 0ULL);
    thrust::device_vector<unsigned long long> list_counter(1, 0ULL);
    thrust::device_vector<std::int32_t> emergency_flags(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> emergency_positions(static_cast<std::size_t>(n));

    thrust::copy(seed_vertices.begin(), seed_vertices.end(), frontier.begin());
    std::uint64_t assigned = static_cast<std::uint64_t>(stats.target_clusters);
    std::int64_t frontier_count = stats.target_clusters;
    std::int32_t epoch = 1;
    auto mark_neighbors = [&](std::int64_t count,
                              const thrust::device_vector<std::int32_t>& source,
                              thrust::device_vector<std::int32_t>& destination) {
        if (count <= 0) return;
        const int blocks = static_cast<int>(
            (count + BASC_WARPS_PER_BLOCK - 1) / BASC_WARPS_PER_BLOCK);
        frontier_mark_neighbors_kernel<<<blocks, 256>>>(
            count, thrust::raw_pointer_cast(source.data()),
            thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(labels.data()), epoch,
            thrust::raw_pointer_cast(active_epoch.data()),
            thrust::raw_pointer_cast(destination.data()),
            thrust::raw_pointer_cast(list_counter.data()));
        CUDA_CHECK(cudaGetLastError());
    };
    auto read_list_count = [&]() {
        unsigned long long count = 0;
        CUDA_CHECK(cudaMemcpy(&count, thrust::raw_pointer_cast(list_counter.data()),
                              sizeof(count), cudaMemcpyDeviceToHost));
        return static_cast<std::int64_t>(count);
    };
    CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(list_counter.data()), 0,
                          sizeof(unsigned long long)));
    mark_neighbors(frontier_count, frontier, active);
    std::int64_t active_count = read_list_count();

    const auto growth_start = std::chrono::steady_clock::now();
    int stalled_rounds = 0;
    int rounds_since_emergency = 0;
    auto create_emergency_seeds = [&]() -> std::int64_t {
        frontier_emergency_mark_kernel<<<warp_blocks, 256>>>(
            n, thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(emergency_priority.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(emergency_flags.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::exclusive_scan(
            emergency_flags.begin(), emergency_flags.end(),
            emergency_positions.begin());
        std::int32_t last_position = 0;
        std::int32_t last_flag = 0;
        CUDA_CHECK(cudaMemcpy(&last_position,
            thrust::raw_pointer_cast(emergency_positions.data()) + (n - 1),
            sizeof(last_position), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_flag,
            thrust::raw_pointer_cast(emergency_flags.data()) + (n - 1),
            sizeof(last_flag), cudaMemcpyDeviceToHost));
        const auto count = static_cast<std::int64_t>(last_position + last_flag);
        if (count <= 0) return 0;
        frontier_compact_flags_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(emergency_flags.data()),
            thrust::raw_pointer_cast(emergency_positions.data()),
            thrust::raw_pointer_cast(frontier.data()));
        CUDA_CHECK(cudaGetLastError());
        const int blocks = static_cast<int>((count + 255) / 256);
        frontier_initialize_seeds_kernel<<<blocks, 256>>>(
            count, thrust::raw_pointer_cast(frontier.data()),
            thrust::raw_pointer_cast(graph.vertex_weights.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(seed_mask.data()),
            thrust::raw_pointer_cast(cluster_weights.data()));
        CUDA_CHECK(cudaGetLastError());
        stats.emergency_seeds += static_cast<std::uint64_t>(count);
        assigned += static_cast<std::uint64_t>(count);
        return count;
    };

    while (assigned < static_cast<std::uint64_t>(n)) {
        // A capacity-saturated label can keep a very thin frontier alive for
        // hundreds of hops.  Re-seed the remaining induced graph after a
        // bounded wave so work stays close to O(V+E) in practice.
        if (active_count == 0 || rounds_since_emergency >= 64) {
            frontier_count = create_emergency_seeds();
            if (frontier_count <= 0) {
                throw std::runtime_error("frontier growth left unreachable vertices");
            }
            CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(list_counter.data()), 0,
                                  sizeof(unsigned long long)));
            ++epoch;
            mark_neighbors(frontier_count, frontier, active);
            active_count = read_list_count();
            stalled_rounds = 0;
            rounds_since_emergency = 0;
            continue;
        }

        ++stats.growth_rounds;
        ++rounds_since_emergency;
        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(proposal_counters.data()), 0,
                              2 * sizeof(unsigned long long)));
        const int active_warp_blocks = static_cast<int>(
            (active_count + BASC_WARPS_PER_BLOCK - 1) / BASC_WARPS_PER_BLOCK);
        frontier_growth_propose_kernel<<<active_warp_blocks, 256>>>(
            active_count, thrust::raw_pointer_cast(active.data()),
            thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(graph.edge_weights.data()),
            thrust::raw_pointer_cast(graph.vertex_weights.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(cluster_weights.data()), stats.cluster_cap,
            confidence_threshold, max_postpone,
            thrust::raw_pointer_cast(postpone.data()),
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(proposal_scores.data()),
            thrust::raw_pointer_cast(proposal_counters.data()),
            thrust::raw_pointer_cast(proposal_counters.data()) + 1);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long round_proposed = 0;
        unsigned long long round_postponed = 0;
        CUDA_CHECK(cudaMemcpy(&round_proposed,
            thrust::raw_pointer_cast(proposal_counters.data()),
            sizeof(round_proposed), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&round_postponed,
            thrust::raw_pointer_cast(proposal_counters.data()) + 1,
            sizeof(round_postponed), cudaMemcpyDeviceToHost));
        stats.proposed += round_proposed;
        stats.postponed += round_postponed;

        thrust::copy_n(active.begin(), active_count, order.begin());
        thrust::sort(
            thrust::device, order.begin(), order.begin() + active_count,
            FrontierProposalOrder{
                thrust::raw_pointer_cast(proposals.data()),
                thrust::raw_pointer_cast(proposal_scores.data())});
        const auto valid_count = static_cast<std::int64_t>(thrust::count_if(
            thrust::device, order.begin(), order.begin() + active_count,
            FrontierHasProposal{thrust::raw_pointer_cast(proposals.data())}));
        std::int64_t accepted_count = 0;
        unsigned long long round_rejected = 0;
        if (valid_count > 0) {
            const int blocks = static_cast<int>((valid_count + 255) / 256);
            frontier_ordered_proposal_data_kernel<<<blocks, 256>>>(
                valid_count, thrust::raw_pointer_cast(order.data()),
                thrust::raw_pointer_cast(proposals.data()),
                thrust::raw_pointer_cast(graph.vertex_weights.data()),
                thrust::raw_pointer_cast(ordered_targets.data()),
                thrust::raw_pointer_cast(ordered_weights.data()));
            CUDA_CHECK(cudaGetLastError());
            thrust::inclusive_scan_by_key(
                thrust::device, ordered_targets.begin(),
                ordered_targets.begin() + valid_count, ordered_weights.begin(),
                prefix_weights.begin());
            thrust::copy(
                cluster_weights.begin(), cluster_weights.end(),
                base_cluster_weights.begin());
            CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(commit_counters.data()), 0,
                                  2 * sizeof(unsigned long long)));
            frontier_commit_prefix_kernel<<<blocks, 256>>>(
                valid_count, thrust::raw_pointer_cast(order.data()),
                thrust::raw_pointer_cast(ordered_targets.data()),
                thrust::raw_pointer_cast(ordered_weights.data()),
                thrust::raw_pointer_cast(prefix_weights.data()),
                thrust::raw_pointer_cast(base_cluster_weights.data()),
                stats.cluster_cap, thrust::raw_pointer_cast(labels.data()),
                thrust::raw_pointer_cast(cluster_weights.data()),
                thrust::raw_pointer_cast(accepted_frontier.data()),
                thrust::raw_pointer_cast(commit_counters.data()),
                thrust::raw_pointer_cast(commit_counters.data()) + 1);
            CUDA_CHECK(cudaGetLastError());
            unsigned long long host_accepted = 0;
            CUDA_CHECK(cudaMemcpy(&host_accepted,
                thrust::raw_pointer_cast(commit_counters.data()),
                sizeof(host_accepted), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&round_rejected,
                thrust::raw_pointer_cast(commit_counters.data()) + 1,
                sizeof(round_rejected), cudaMemcpyDeviceToHost));
            accepted_count = static_cast<std::int64_t>(host_accepted);
        }
        stats.accepted += static_cast<std::uint64_t>(accepted_count);
        stats.capacity_rejected += round_rejected;
        assigned += static_cast<std::uint64_t>(accepted_count);

        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(list_counter.data()), 0,
                              sizeof(unsigned long long)));
        ++epoch;
        const int active_blocks = static_cast<int>((active_count + 255) / 256);
        frontier_carry_unassigned_kernel<<<active_blocks, 256>>>(
            active_count, thrust::raw_pointer_cast(active.data()),
            thrust::raw_pointer_cast(labels.data()), epoch,
            thrust::raw_pointer_cast(active_epoch.data()),
            thrust::raw_pointer_cast(next_active.data()),
            thrust::raw_pointer_cast(list_counter.data()));
        CUDA_CHECK(cudaGetLastError());
        if (accepted_count > 0) {
            mark_neighbors(accepted_count, accepted_frontier, next_active);
        }
        auto next_count = read_list_count();
        if (diagnostics) {
            std::cout << "ml_gpu_frontier_round level=" << level
                      << " round=" << stats.growth_rounds
                      << " active=" << active_count
                      << " proposed=" << round_proposed
                      << " postponed=" << round_postponed
                      << " accepted=" << accepted_count
                      << " capacity_rejected=" << round_rejected
                      << " assigned=" << assigned << '\n';
        }
        if (accepted_count == 0) {
            ++stalled_rounds;
            if (round_postponed == 0 &&
                (round_proposed == 0 || stalled_rounds >= 2)) {
                frontier_count = create_emergency_seeds();
                if (frontier_count <= 0) {
                    throw std::runtime_error("frontier growth stalled without an emergency seed");
                }
                CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(list_counter.data()), 0,
                                      sizeof(unsigned long long)));
                ++epoch;
                mark_neighbors(frontier_count, frontier, active);
                active_count = read_list_count();
                stalled_rounds = 0;
                rounds_since_emergency = 0;
                continue;
            }
        } else {
            stalled_rounds = 0;
        }
        active.swap(next_active);
        active_count = next_count;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.growth_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - growth_start).count();

    const auto boundary_start = std::chrono::steady_clock::now();
    thrust::device_vector<std::uint64_t> boundary_gains(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint8_t> boundary_winners(static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> boundary_counters(2, 0ULL);
    thrust::device_vector<unsigned long long> cut_counter(1, 0ULL);
    auto device_label_cut = [&]() {
        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(cut_counter.data()), 0,
                              sizeof(unsigned long long)));
        weighted_cut_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(graph.edge_weights.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(cut_counter.data()));
        CUDA_CHECK(cudaGetLastError());
        unsigned long long directed = 0;
        CUDA_CHECK(cudaMemcpy(&directed,
            thrust::raw_pointer_cast(cut_counter.data()), sizeof(directed),
            cudaMemcpyDeviceToHost));
        if (directed & 1ULL) {
            throw std::runtime_error("frontier boundary cut is not symmetric");
        }
        return static_cast<std::uint64_t>(directed / 2);
    };
    for (int boundary_round = 0; boundary_round < 2; ++boundary_round) {
        const auto cut_before = diagnostics ? device_label_cut() : 0;
        frontier_boundary_propose_kernel<<<warp_blocks, 256>>>(
            n, thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(graph.edge_weights.data()),
            thrust::raw_pointer_cast(graph.vertex_weights.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(seed_mask.data()),
            thrust::raw_pointer_cast(cluster_weights.data()), stats.cluster_cap,
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(proposal_scores.data()),
            thrust::raw_pointer_cast(boundary_gains.data()));
        CUDA_CHECK(cudaGetLastError());
        // The top-K support table chooses a bounded target candidate.  Re-scan
        // that one target exactly before conflict selection: on coarse graphs a
        // vertex can touch more than K labels, so the bounded table is not an
        // exact gain accumulator.
        frontier_boundary_exact_gain_kernel<<<warp_blocks, 256>>>(
            n, thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(graph.edge_weights.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(proposal_scores.data()),
            thrust::raw_pointer_cast(boundary_gains.data()));
        CUDA_CHECK(cudaGetLastError());
        frontier_boundary_conflict_kernel<<<warp_blocks, 256>>>(
            n, thrust::raw_pointer_cast(graph.offsets.data()),
            thrust::raw_pointer_cast(graph.neighbors.data()),
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(boundary_gains.data()),
            thrust::raw_pointer_cast(boundary_winners.data()));
        CUDA_CHECK(cudaGetLastError());
        frontier_apply_boundary_conflicts_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(boundary_winners.data()),
            thrust::raw_pointer_cast(proposals.data()));
        CUDA_CHECK(cudaGetLastError());

        thrust::sequence(order.begin(), order.end());
        thrust::sort(
            thrust::device, order.begin(), order.end(),
            FrontierProposalOrder{
                thrust::raw_pointer_cast(proposals.data()),
                thrust::raw_pointer_cast(proposal_scores.data())});
        const auto valid_count = static_cast<std::int64_t>(thrust::count_if(
            thrust::device, order.begin(), order.end(),
            FrontierHasProposal{thrust::raw_pointer_cast(proposals.data())}));
        if (valid_count == 0) break;
        const int blocks = static_cast<int>((valid_count + 255) / 256);
        frontier_ordered_proposal_data_kernel<<<blocks, 256>>>(
            valid_count, thrust::raw_pointer_cast(order.data()),
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(graph.vertex_weights.data()),
            thrust::raw_pointer_cast(ordered_targets.data()),
            thrust::raw_pointer_cast(ordered_weights.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::inclusive_scan_by_key(
            thrust::device, ordered_targets.begin(),
            ordered_targets.begin() + valid_count, ordered_weights.begin(),
            prefix_weights.begin());
        thrust::copy(cluster_weights.begin(), cluster_weights.end(),
                     base_cluster_weights.begin());
        CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(boundary_counters.data()), 0,
                              2 * sizeof(unsigned long long)));
        frontier_boundary_commit_kernel<<<blocks, 256>>>(
            valid_count, thrust::raw_pointer_cast(order.data()),
            thrust::raw_pointer_cast(ordered_targets.data()),
            thrust::raw_pointer_cast(ordered_weights.data()),
            thrust::raw_pointer_cast(prefix_weights.data()),
            thrust::raw_pointer_cast(base_cluster_weights.data()),
            stats.cluster_cap, thrust::raw_pointer_cast(boundary_gains.data()),
            thrust::raw_pointer_cast(labels.data()),
            thrust::raw_pointer_cast(cluster_weights.data()),
            thrust::raw_pointer_cast(boundary_counters.data()),
            thrust::raw_pointer_cast(boundary_counters.data()) + 1);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long moved = 0;
        unsigned long long gain = 0;
        CUDA_CHECK(cudaMemcpy(&moved,
            thrust::raw_pointer_cast(boundary_counters.data()), sizeof(moved),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&gain,
            thrust::raw_pointer_cast(boundary_counters.data()) + 1, sizeof(gain),
            cudaMemcpyDeviceToHost));
        ++stats.boundary_rounds;
        stats.boundary_moved += moved;
        stats.boundary_gain += gain;
        if (diagnostics) {
            const auto cut_after = device_label_cut();
            if (cut_after > cut_before || cut_before - cut_after != gain) {
                throw std::runtime_error(
                    "frontier boundary gain does not match full cut recomputation");
            }
            std::cout << "ml_gpu_frontier_boundary level=" << level
                      << " round=" << boundary_round
                      << " proposed_nonconflicting=" << valid_count
                      << " moved=" << moved
                      << " gain=" << gain
                      << " cut_before=" << cut_before
                      << " cut_after=" << cut_after << '\n';
        }
        if (moved == 0) break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.boundary_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - boundary_start).count();

    thrust::device_vector<std::int32_t> root_flags(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> root_ids(static_cast<std::size_t>(n));
    frontier_mark_label_roots_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(labels.data()),
        thrust::raw_pointer_cast(root_flags.data()));
    CUDA_CHECK(cudaGetLastError());
    thrust::exclusive_scan(root_flags.begin(), root_flags.end(), root_ids.begin());
    std::int32_t last_root_id = 0;
    std::int32_t last_root_flag = 0;
    CUDA_CHECK(cudaMemcpy(&last_root_id,
        thrust::raw_pointer_cast(root_ids.data()) + (n - 1),
        sizeof(last_root_id), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&last_root_flag,
        thrust::raw_pointer_cast(root_flags.data()) + (n - 1),
        sizeof(last_root_flag), cudaMemcpyDeviceToHost));

    DeviceAggregateResult out;
    out.coarse_vertices = last_root_id + last_root_flag;
    out.capacity = stats.cluster_cap;
    if (out.coarse_vertices <= 0 ||
        out.coarse_vertices != stats.target_clusters +
            static_cast<std::int64_t>(stats.emergency_seeds)) {
        throw std::runtime_error("frontier coarsening root count mismatch");
    }
    out.map.resize(static_cast<std::size_t>(n));
    frontier_compact_labels_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(labels.data()),
        thrust::raw_pointer_cast(root_ids.data()),
        thrust::raw_pointer_cast(out.map.data()));
    CUDA_CHECK(cudaGetLastError());
    out.vertex_weights.resize(static_cast<std::size_t>(out.coarse_vertices));
    thrust::fill(out.vertex_weights.begin(), out.vertex_weights.end(), std::uint64_t{0});
    basc_coarse_vertex_weights_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.vertex_weights.data()),
        thrust::raw_pointer_cast(out.map.data()),
        thrust::raw_pointer_cast(out.vertex_weights.data()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    out.maximum_weight = thrust::reduce(
        out.vertex_weights.begin(), out.vertex_weights.end(),
        std::uint64_t{0}, thrust::maximum<std::uint64_t>());
    if (out.maximum_weight > stats.cluster_cap) {
        throw std::runtime_error("frontier coarsening exceeded cluster capacity");
    }
    std::cout << "ml_gpu_frontier level=" << level
              << " target_clusters=" << stats.target_clusters
              << " coarse_vertices=" << out.coarse_vertices
              << " hot_seeds=" << stats.hot_seeds
              << " cold_seeds=" << stats.cold_seeds
              << " cold_seed_rounds=" << stats.cold_seed_rounds
              << " emergency_seeds=" << stats.emergency_seeds
              << " cluster_cap=" << stats.cluster_cap
              << " capacity_slack=" << capacity_slack
              << " contraction_factor=" << contraction_factor
              << " hot_ratio=" << hot_ratio
              << " growth_rounds=" << stats.growth_rounds
              << " proposed=" << stats.proposed
              << " postponed=" << stats.postponed
              << " accepted=" << stats.accepted
              << " capacity_rejected=" << stats.capacity_rejected
              << " seed_seconds=" << stats.seed_seconds
              << " growth_seconds=" << stats.growth_seconds
              << " boundary_rounds=" << stats.boundary_rounds
              << " boundary_moved=" << stats.boundary_moved
              << " boundary_gain=" << stats.boundary_gain
              << " boundary_seconds=" << stats.boundary_seconds << '\n';
    return out;
}

DeviceAggregateResult gpu_sclp_aggregate_device(
    const DeviceWeightedGraph& graph, int parts, std::uint32_t seed,
    int level, SclpStats& stats, bool diagnostics) {
    const auto n = graph.vertices();
    const auto m = graph.edges();
    if (n <= 0 || n > std::numeric_limits<std::int32_t>::max()) {
        throw std::runtime_error("SCLP coarsening received an unsupported graph size");
    }
    const auto read_environment_integer = [](
        const char* name, int fallback, int minimum, int maximum) {
        const char* raw = std::getenv(name);
        if (raw == nullptr) return fallback;
        std::size_t consumed = 0;
        const int value = std::stoi(raw, &consumed);
        if (raw[consumed] != '\0' || value < minimum || value > maximum) {
            throw std::runtime_error(std::string("invalid ") + name);
        }
        return value;
    };
    const int beta = read_environment_integer("SCLP_BETA", 64, 1, 4096);
    const int maximum_rounds = read_environment_integer("SCLP_ROUNDS", 4, 2, 4);
    constexpr double required_ratio = 0.50;
    const auto total_weight = thrust::reduce(
        graph.vertex_weights.begin(), graph.vertex_weights.end(),
        std::uint64_t{0}, thrust::plus<std::uint64_t>());
    const auto maximum_vertex = thrust::reduce(
        graph.vertex_weights.begin(), graph.vertex_weights.end(),
        std::uint64_t{0}, thrust::maximum<std::uint64_t>());
    const auto denominator = static_cast<std::uint64_t>(beta) *
                             static_cast<std::uint64_t>(parts);
    const auto average_cap = total_weight / denominator +
                             (total_weight % denominator != 0 ? 1 : 0);
    stats.capacity = std::max(maximum_vertex, average_cap);

    const int vertex_blocks = static_cast<int>((n + 255) / 256);
    thrust::device_vector<std::int32_t> clusters(static_cast<std::size_t>(n));
    thrust::sequence(clusters.begin(), clusters.end());
    thrust::device_vector<unsigned long long> cluster_weights(
        graph.vertex_weights.begin(), graph.vertex_weights.end());

    // Degree classes are fixed for this graph level and built once.  Low
    // vertices are handled thread-per-vertex (many per warp), medium vertices
    // warp-per-vertex, and high vertices CTA-per-vertex.
    thrust::device_vector<std::int32_t> low_flags(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> medium_flags(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> high_flags(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> positions(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> low_vertices(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> medium_vertices(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> high_vertices(static_cast<std::size_t>(n));
    sclp_degree_class_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(graph.offsets.data()),
        thrust::raw_pointer_cast(low_flags.data()),
        thrust::raw_pointer_cast(medium_flags.data()),
        thrust::raw_pointer_cast(high_flags.data()));
    CUDA_CHECK(cudaGetLastError());
    const auto compact_class = [&](const thrust::device_vector<std::int32_t>& flags,
                                   thrust::device_vector<std::int32_t>& vertices) {
        thrust::exclusive_scan(flags.begin(), flags.end(), positions.begin());
        std::int32_t last_position = 0;
        std::int32_t last_flag = 0;
        CUDA_CHECK(cudaMemcpy(&last_position,
            thrust::raw_pointer_cast(positions.data()) + (n - 1),
            sizeof(last_position), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last_flag,
            thrust::raw_pointer_cast(flags.data()) + (n - 1),
            sizeof(last_flag), cudaMemcpyDeviceToHost));
        frontier_compact_flags_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(flags.data()),
            thrust::raw_pointer_cast(positions.data()),
            thrust::raw_pointer_cast(vertices.data()));
        CUDA_CHECK(cudaGetLastError());
        return static_cast<std::int64_t>(last_position + last_flag);
    };
    const auto low_count = compact_class(low_flags, low_vertices);
    const auto medium_count = compact_class(medium_flags, medium_vertices);
    const auto high_count = compact_class(high_flags, high_vertices);

    thrust::device_vector<std::uint64_t> affinity_keys(static_cast<std::size_t>(m));
    thrust::device_vector<std::uint64_t> affinity_values(static_cast<std::size_t>(m));
    thrust::device_vector<std::uint64_t> unique_keys(static_cast<std::size_t>(m));
    thrust::device_vector<std::uint64_t> unique_values(static_cast<std::size_t>(m));
    thrust::device_vector<unsigned long long> best_affinity(
        static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> current_affinity(
        static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> best_ties(
        static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> gains(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint32_t> proposal_ties(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> proposals(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> order(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> ordered_targets(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint64_t> ordered_weights(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint64_t> prefix_weights(static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> base_cluster_weights(
        static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> counters(3, 0ULL);
    thrust::device_vector<unsigned long long> cut_counter(1, 0ULL);
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
        int round, bool filter_roles, std::uint32_t mover_threshold) {
        const auto start = std::chrono::steady_clock::now();
        if (low_count > 0) {
            const int blocks = static_cast<int>((low_count + 255) / 256);
            sclp_fill_affinity_low_kernel<<<blocks, 256>>>(
                low_count, thrust::raw_pointer_cast(low_vertices.data()),
                thrust::raw_pointer_cast(graph.offsets.data()),
                thrust::raw_pointer_cast(graph.neighbors.data()),
                thrust::raw_pointer_cast(graph.edge_weights.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(affinity_keys.data()),
                thrust::raw_pointer_cast(affinity_values.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        if (medium_count > 0) {
            const int blocks = static_cast<int>(
                (medium_count + BASC_WARPS_PER_BLOCK - 1) / BASC_WARPS_PER_BLOCK);
            sclp_fill_affinity_medium_kernel<<<blocks, 256>>>(
                medium_count, thrust::raw_pointer_cast(medium_vertices.data()),
                thrust::raw_pointer_cast(graph.offsets.data()),
                thrust::raw_pointer_cast(graph.neighbors.data()),
                thrust::raw_pointer_cast(graph.edge_weights.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(affinity_keys.data()),
                thrust::raw_pointer_cast(affinity_values.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        if (high_count > 0) {
            sclp_fill_affinity_high_kernel<<<static_cast<int>(high_count), 256>>>(
                high_count, thrust::raw_pointer_cast(high_vertices.data()),
                thrust::raw_pointer_cast(graph.offsets.data()),
                thrust::raw_pointer_cast(graph.neighbors.data()),
                thrust::raw_pointer_cast(graph.edge_weights.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(affinity_keys.data()),
                thrust::raw_pointer_cast(affinity_values.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        thrust::sort_by_key(
            thrust::device, affinity_keys.begin(), affinity_keys.end(),
            affinity_values.begin());
        const auto valid_end = thrust::lower_bound(
            thrust::device, affinity_keys.begin(), affinity_keys.end(),
            BASC_INVALID_KEY);
        const auto reduced = thrust::reduce_by_key(
            thrust::device, affinity_keys.begin(), valid_end,
            affinity_values.begin(), unique_keys.begin(), unique_values.begin());
        const auto unique_count = static_cast<std::int64_t>(
            reduced.first - unique_keys.begin());
        thrust::fill(best_affinity.begin(), best_affinity.end(), 0ULL);
        thrust::fill(current_affinity.begin(), current_affinity.end(), 0ULL);
        thrust::fill(best_ties.begin(), best_ties.end(), 0ULL);
        const auto role_salt = seed ^
            (static_cast<std::uint32_t>(level) * 0x9e3779b9U) ^
            (static_cast<std::uint32_t>(round) * 0x85ebca6bU) ^ 0x27d4eb2dU;
        const auto tie_salt = seed ^
            (static_cast<std::uint32_t>(level) * 0xc2b2ae35U) ^
            (static_cast<std::uint32_t>(round) * 0x165667b1U) ^ 0xd3a2646cU;
        if (unique_count > 0) {
            const int blocks = static_cast<int>((unique_count + 255) / 256);
            sclp_affinity_baseline_kernel<<<blocks, 256>>>(
                unique_count, thrust::raw_pointer_cast(unique_keys.data()),
                thrust::raw_pointer_cast(unique_values.data()),
                thrust::raw_pointer_cast(clusters.data()),
                role_salt, mover_threshold, filter_roles,
                thrust::raw_pointer_cast(current_affinity.data()),
                thrust::raw_pointer_cast(best_affinity.data()));
            CUDA_CHECK(cudaGetLastError());
            sclp_best_tie_kernel<<<blocks, 256>>>(
                unique_count, thrust::raw_pointer_cast(unique_keys.data()),
                thrust::raw_pointer_cast(unique_values.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(best_affinity.data()),
                tie_salt, role_salt, mover_threshold, filter_roles,
                thrust::raw_pointer_cast(best_ties.data()));
            CUDA_CHECK(cudaGetLastError());
        }
        sclp_decode_gain_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(clusters.data()),
            thrust::raw_pointer_cast(current_affinity.data()),
            thrust::raw_pointer_cast(best_affinity.data()),
            thrust::raw_pointer_cast(best_ties.data()),
            role_salt, mover_threshold, filter_roles,
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(gains.data()),
            thrust::raw_pointer_cast(proposal_ties.data()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        stats.affinity_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
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
        form_exact_proposals(round, true, mover_threshold);
        const auto admission_start = std::chrono::steady_clock::now();
        const auto cut_before = diagnostics ? device_cluster_cut() : 0;
        thrust::sequence(order.begin(), order.end());
        thrust::sort(
            thrust::device, order.begin(), order.end(),
            SclpProposalOrder{
                thrust::raw_pointer_cast(proposals.data()),
                thrust::raw_pointer_cast(gains.data()),
                thrust::raw_pointer_cast(proposal_ties.data())});
        const auto valid_count = static_cast<std::int64_t>(thrust::count_if(
            thrust::device, order.begin(), order.end(),
            SclpHasProposal{thrust::raw_pointer_cast(proposals.data())}));
        unsigned long long accepted = 0;
        unsigned long long rejected = 0;
        unsigned long long predicted_gain = 0;
        if (valid_count > 0) {
            const int blocks = static_cast<int>((valid_count + 255) / 256);
            frontier_ordered_proposal_data_kernel<<<blocks, 256>>>(
                valid_count, thrust::raw_pointer_cast(order.data()),
                thrust::raw_pointer_cast(proposals.data()),
                thrust::raw_pointer_cast(graph.vertex_weights.data()),
                thrust::raw_pointer_cast(ordered_targets.data()),
                thrust::raw_pointer_cast(ordered_weights.data()));
            CUDA_CHECK(cudaGetLastError());
            thrust::inclusive_scan_by_key(
                thrust::device, ordered_targets.begin(),
                ordered_targets.begin() + valid_count, ordered_weights.begin(),
                prefix_weights.begin());
            thrust::copy(cluster_weights.begin(), cluster_weights.end(),
                         base_cluster_weights.begin());
            CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counters.data()), 0,
                                  3 * sizeof(unsigned long long)));
            sclp_commit_kernel<<<blocks, 256>>>(
                valid_count, thrust::raw_pointer_cast(order.data()),
                thrust::raw_pointer_cast(ordered_targets.data()),
                thrust::raw_pointer_cast(ordered_weights.data()),
                thrust::raw_pointer_cast(prefix_weights.data()),
                thrust::raw_pointer_cast(base_cluster_weights.data()),
                stats.capacity, thrust::raw_pointer_cast(gains.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(cluster_weights.data()),
                thrust::raw_pointer_cast(counters.data()),
                thrust::raw_pointer_cast(counters.data()) + 1,
                thrust::raw_pointer_cast(counters.data()) + 2);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&accepted,
                thrust::raw_pointer_cast(counters.data()), sizeof(accepted),
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&rejected,
                thrust::raw_pointer_cast(counters.data()) + 1, sizeof(rejected),
                cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&predicted_gain,
                thrust::raw_pointer_cast(counters.data()) + 2,
                sizeof(predicted_gain), cudaMemcpyDeviceToHost));
        }
        cluster_count = static_cast<std::int64_t>(thrust::count_if(
            cluster_weights.begin(), cluster_weights.end(), SclpNonzeroWeight{}));
        stats.lp_accepted += accepted;
        stats.capacity_rejected += rejected;
        ++stats.rounds;
        stats.admission_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - admission_start).count();
        const auto cut_after = diagnostics ? device_cluster_cut() : 0;
        if (diagnostics && cut_after > cut_before) {
            throw std::runtime_error("SCLP synchronous batch increased cluster cut");
        }
        std::cout << "ml_gpu_sclp_round level=" << level
                  << " round=" << round
                  << " proposals=" << valid_count
                  << " accepted=" << accepted
                  << " capacity_rejected=" << rejected
                  << " predicted_gain=" << predicted_gain;
        if (diagnostics) {
            std::cout << " cut_before=" << cut_before
                      << " cut_after=" << cut_after
                      << " actual_gain=" << (cut_before - cut_after);
        }
        std::cout
                  << " mover_fraction=" << mover_fraction
                  << " clusters=" << cluster_count
                  << " contraction_ratio="
                  << static_cast<double>(cluster_count) / static_cast<double>(n)
                  << '\n';
        if (static_cast<double>(cluster_count) <= 0.60 * n) break;
    }

    // Simple two-hop fallback: remaining singleton vertices with the same
    // one-hop favorite form at most one deterministic capacity-bounded group.
    // This is a fallback, not part of the SCLP quality claim.
    if (static_cast<double>(cluster_count) > required_ratio * n) {
        const auto fallback_start = std::chrono::steady_clock::now();
        const auto fallback_cut_before = diagnostics ? device_cluster_cut() : 0;
        form_exact_proposals(maximum_rounds, false, 0);
        thrust::device_vector<std::int32_t> favorites(static_cast<std::size_t>(n));
        sclp_singleton_favorite_kernel<<<vertex_blocks, 256>>>(
            n, thrust::raw_pointer_cast(graph.vertex_weights.data()),
            thrust::raw_pointer_cast(cluster_weights.data()),
            thrust::raw_pointer_cast(clusters.data()),
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(favorites.data()));
        CUDA_CHECK(cudaGetLastError());
        thrust::sequence(order.begin(), order.end());
        thrust::sort(
            thrust::device, order.begin(), order.end(),
            SclpProposalOrder{
                thrust::raw_pointer_cast(favorites.data()),
                thrust::raw_pointer_cast(gains.data()),
                thrust::raw_pointer_cast(proposal_ties.data())});
        const auto singleton_candidates = static_cast<std::int64_t>(thrust::count_if(
            thrust::device, order.begin(), order.end(),
            SclpHasProposal{thrust::raw_pointer_cast(favorites.data())}));
        if (singleton_candidates > 0) {
            const int blocks = static_cast<int>((singleton_candidates + 255) / 256);
            frontier_ordered_proposal_data_kernel<<<blocks, 256>>>(
                singleton_candidates, thrust::raw_pointer_cast(order.data()),
                thrust::raw_pointer_cast(favorites.data()),
                thrust::raw_pointer_cast(graph.vertex_weights.data()),
                thrust::raw_pointer_cast(ordered_targets.data()),
                thrust::raw_pointer_cast(ordered_weights.data()));
            CUDA_CHECK(cudaGetLastError());
            thrust::device_vector<std::uint64_t> pair_ones(
                static_cast<std::size_t>(singleton_candidates), 1);
            thrust::device_vector<std::uint64_t> pair_ranks(
                static_cast<std::size_t>(singleton_candidates));
            thrust::inclusive_scan_by_key(
                thrust::device, ordered_targets.begin(),
                ordered_targets.begin() + singleton_candidates,
                pair_ones.begin(), pair_ranks.begin());
            thrust::device_vector<std::int32_t> pair_flags(
                static_cast<std::size_t>(singleton_candidates));
            thrust::device_vector<std::int32_t> pair_positions(
                static_cast<std::size_t>(singleton_candidates));
            sclp_pair_flags_kernel<<<blocks, 256>>>(
                singleton_candidates, thrust::raw_pointer_cast(order.data()),
                thrust::raw_pointer_cast(ordered_weights.data()),
                thrust::raw_pointer_cast(pair_ranks.data()), stats.capacity,
                thrust::raw_pointer_cast(pair_flags.data()));
            CUDA_CHECK(cudaGetLastError());
            thrust::exclusive_scan(
                pair_flags.begin(), pair_flags.end(), pair_positions.begin());
            const auto merge_budget = static_cast<std::uint64_t>(
                std::max<std::int64_t>(0, cluster_count - desired_clusters));
            CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counters.data()), 0,
                                  sizeof(unsigned long long)));
            sclp_two_hop_pair_commit_kernel<<<blocks, 256>>>(
                singleton_candidates, thrust::raw_pointer_cast(order.data()),
                thrust::raw_pointer_cast(pair_flags.data()),
                thrust::raw_pointer_cast(pair_positions.data()), merge_budget,
                thrust::raw_pointer_cast(ordered_weights.data()),
                thrust::raw_pointer_cast(clusters.data()),
                thrust::raw_pointer_cast(cluster_weights.data()),
                thrust::raw_pointer_cast(counters.data()));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&stats.two_hop_merged,
                thrust::raw_pointer_cast(counters.data()),
                sizeof(stats.two_hop_merged), cudaMemcpyDeviceToHost));
        }
        cluster_count = static_cast<std::int64_t>(thrust::count_if(
            cluster_weights.begin(), cluster_weights.end(), SclpNonzeroWeight{}));
        if (diagnostics) {
            const auto fallback_cut_after = device_cluster_cut();
            if (fallback_cut_after > fallback_cut_before) {
                throw std::runtime_error("SCLP two-hop pairing increased cluster cut");
            }
            std::cout << "ml_gpu_sclp_two_hop level=" << level
                      << " merged=" << stats.two_hop_merged
                      << " cut_before=" << fallback_cut_before
                      << " cut_after=" << fallback_cut_after
                      << " actual_gain="
                      << (fallback_cut_before - fallback_cut_after) << '\n';
        }
        stats.two_hop_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - fallback_start).count();
    }

    thrust::device_vector<std::int32_t> nonempty_flags(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> compact_ids(static_cast<std::size_t>(n));
    sclp_nonempty_cluster_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(cluster_weights.data()),
        thrust::raw_pointer_cast(nonempty_flags.data()));
    CUDA_CHECK(cudaGetLastError());
    thrust::exclusive_scan(nonempty_flags.begin(), nonempty_flags.end(),
                           compact_ids.begin());

    DeviceAggregateResult out;
    out.coarse_vertices = static_cast<std::int32_t>(cluster_count);
    out.capacity = stats.capacity;
    out.map.resize(static_cast<std::size_t>(n));
    sclp_compact_map_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(clusters.data()),
        thrust::raw_pointer_cast(compact_ids.data()),
        thrust::raw_pointer_cast(out.map.data()));
    CUDA_CHECK(cudaGetLastError());
    out.vertex_weights.resize(static_cast<std::size_t>(out.coarse_vertices));
    thrust::fill(out.vertex_weights.begin(), out.vertex_weights.end(), std::uint64_t{0});
    basc_coarse_vertex_weights_kernel<<<vertex_blocks, 256>>>(
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
    // A singleton cluster has exactly one member; count it from the compact map
    // rather than equating vertex and aggregate weights (weighted vertices may
    // legitimately have equal weights).
    thrust::device_vector<std::uint64_t> member_counts(
        static_cast<std::size_t>(out.coarse_vertices), 0);
    thrust::device_vector<std::uint64_t> unit_weights(static_cast<std::size_t>(n), 1);
    basc_coarse_vertex_weights_kernel<<<vertex_blocks, 256>>>(
        n, thrust::raw_pointer_cast(unit_weights.data()),
        thrust::raw_pointer_cast(out.map.data()),
        thrust::raw_pointer_cast(member_counts.data()));
    CUDA_CHECK(cudaGetLastError());
    stats.singleton_count = static_cast<std::uint64_t>(thrust::count(
        member_counts.begin(), member_counts.end(), std::uint64_t{1}));
    std::cout << "ml_gpu_sclp level=" << level
              << " fine_vertices=" << n
              << " coarse_vertices=" << out.coarse_vertices
              << " contraction_ratio="
              << static_cast<double>(out.coarse_vertices) / static_cast<double>(n)
              << " beta=" << beta
              << " cluster_cap=" << stats.capacity
              << " lp_rounds=" << stats.rounds
              << " lp_accepted=" << stats.lp_accepted
              << " capacity_rejected=" << stats.capacity_rejected
              << " singleton=" << stats.singleton_count
              << " two_hop_merged=" << stats.two_hop_merged
              << " low_vertices=" << low_count
              << " medium_vertices=" << medium_count
              << " high_vertices=" << high_count
              << " affinity_seconds=" << stats.affinity_seconds
              << " admission_seconds=" << stats.admission_seconds
              << " two_hop_seconds=" << stats.two_hop_seconds
              << " seed=" << seed
              << " diagnostics=" << (diagnostics ? 1 : 0) << '\n';
    return out;
}

DeviceWeightedGraph gpu_contract_graph(
    const DeviceWeightedGraph& fine, const DeviceAggregateResult& aggregate,
    double& seconds) {
    const auto start = std::chrono::steady_clock::now();
    const auto n = fine.vertices();
    const auto m = fine.edges();
    const int warp_blocks = static_cast<int>(
        (n + BASC_WARPS_PER_BLOCK - 1) / BASC_WARPS_PER_BLOCK);
    thrust::device_vector<std::uint64_t> keys(static_cast<std::size_t>(m));
    thrust::device_vector<std::uint64_t> values(fine.edge_weights);
    thrust::device_vector<std::uint64_t> unique_keys(static_cast<std::size_t>(m));
    thrust::device_vector<std::uint64_t> unique_values(static_cast<std::size_t>(m));
    basc_fill_contraction_keys_warp_kernel<<<warp_blocks, 256>>>(
        n, thrust::raw_pointer_cast(fine.offsets.data()),
        thrust::raw_pointer_cast(fine.neighbors.data()),
        thrust::raw_pointer_cast(fine.edge_weights.data()),
        thrust::raw_pointer_cast(aggregate.map.data()),
        thrust::raw_pointer_cast(keys.data()),
        thrust::raw_pointer_cast(values.data()));
    CUDA_CHECK(cudaGetLastError());
    thrust::sort_by_key(thrust::device, keys.begin(), keys.end(), values.begin());
    const auto valid_end = thrust::lower_bound(
        thrust::device, keys.begin(), keys.end(), BASC_INVALID_KEY);
    const auto reduced = thrust::reduce_by_key(
        thrust::device, keys.begin(), valid_end, values.begin(),
        unique_keys.begin(), unique_values.begin());
    const auto coarse_edges = static_cast<std::int64_t>(reduced.first - unique_keys.begin());

    DeviceWeightedGraph coarse;
    coarse.vertex_weights = aggregate.vertex_weights;
    coarse.offsets.resize(static_cast<std::size_t>(aggregate.coarse_vertices) + 1);
    thrust::device_vector<std::int64_t> row_counts(
        static_cast<std::size_t>(aggregate.coarse_vertices) + 1, 0);
    if (coarse_edges > 0) {
        const int blocks = static_cast<int>((coarse_edges + 255) / 256);
        basc_count_coarse_rows_kernel<<<blocks, 256>>>(
            coarse_edges, thrust::raw_pointer_cast(unique_keys.data()),
            aggregate.coarse_vertices, thrust::raw_pointer_cast(row_counts.data()));
        CUDA_CHECK(cudaGetLastError());
    }
    thrust::exclusive_scan(row_counts.begin(), row_counts.end(), coarse.offsets.begin());
    coarse.neighbors.resize(static_cast<std::size_t>(coarse_edges));
    coarse.edge_weights.resize(static_cast<std::size_t>(coarse_edges));
    if (coarse_edges > 0) {
        const int blocks = static_cast<int>((coarse_edges + 255) / 256);
        basc_write_coarse_edges_kernel<<<blocks, 256>>>(
            coarse_edges, thrust::raw_pointer_cast(unique_keys.data()),
            thrust::raw_pointer_cast(unique_values.data()),
            thrust::raw_pointer_cast(coarse.neighbors.data()),
            thrust::raw_pointer_cast(coarse.edge_weights.data()));
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return coarse;
}

WeightedGraph contract_graph(
    const WeightedGraph& fine, const AggregateResult& aggregate) {
    WeightedGraph coarse;
    const auto nc = aggregate.coarse_vertices;
    coarse.vertex_weights.assign(static_cast<std::size_t>(nc), 0);
    for (std::int64_t v = 0; v < fine.vertices(); ++v) {
        coarse.vertex_weights[static_cast<std::size_t>(aggregate.map[static_cast<std::size_t>(v)])] +=
            fine.vertex_weights[static_cast<std::size_t>(v)];
    }
    std::vector<EdgeRecord> records;
    records.reserve(static_cast<std::size_t>(fine.edges()));
    for (std::int64_t v = 0; v < fine.vertices(); ++v) {
        const auto cv = static_cast<std::uint32_t>(aggregate.map[static_cast<std::size_t>(v)]);
        for (auto e = fine.offsets[static_cast<std::size_t>(v)];
             e < fine.offsets[static_cast<std::size_t>(v + 1)]; ++e) {
            const auto cu = static_cast<std::uint32_t>(aggregate.map[
                static_cast<std::size_t>(fine.neighbors[static_cast<std::size_t>(e)])]);
            if (cv == cu) continue;
            records.push_back({(static_cast<std::uint64_t>(cv) << 32) | cu,
                               fine.edge_weights[static_cast<std::size_t>(e)]});
        }
    }
    std::sort(records.begin(), records.end());
    coarse.offsets.assign(static_cast<std::size_t>(nc + 1), 0);
    for (std::size_t begin = 0; begin < records.size();) {
        std::size_t end = begin + 1;
        std::uint64_t weight = records[begin].weight;
        while (end < records.size() && records[end].key == records[begin].key) {
            weight += records[end].weight;
            ++end;
        }
        const auto source = static_cast<std::uint32_t>(records[begin].key >> 32);
        const auto target = static_cast<std::uint32_t>(records[begin].key);
        coarse.neighbors.push_back(static_cast<std::int32_t>(target));
        coarse.edge_weights.push_back(weight);
        ++coarse.offsets[static_cast<std::size_t>(source + 1)];
        begin = end;
    }
    for (std::int32_t v = 0; v < nc; ++v) {
        coarse.offsets[static_cast<std::size_t>(v + 1)] +=
            coarse.offsets[static_cast<std::size_t>(v)];
    }
    return coarse;
}

void report_contraction_metrics(
    const WeightedGraph& fine, const WeightedGraph& coarse,
    const std::vector<std::int32_t>& map, int level) {
    std::uint64_t total_edge_weight = 0;
    std::uint64_t internal_edge_weight = 0;
    for (std::int64_t v = 0; v < fine.vertices(); ++v) {
        for (auto e = fine.offsets[static_cast<std::size_t>(v)];
             e < fine.offsets[static_cast<std::size_t>(v + 1)]; ++e) {
            const auto weight = fine.edge_weights[static_cast<std::size_t>(e)];
            total_edge_weight += weight;
            if (map[static_cast<std::size_t>(v)] == map[static_cast<std::size_t>(
                    fine.neighbors[static_cast<std::size_t>(e)])]) {
                internal_edge_weight += weight;
            }
        }
    }
    std::vector<std::uint64_t> sizes = coarse.vertex_weights;
    std::sort(sizes.begin(), sizes.end());
    const auto percentile = [&](double fraction) {
        if (sizes.empty()) return std::uint64_t{0};
        const auto index = std::min<std::size_t>(
            sizes.size() - 1,
            static_cast<std::size_t>(fraction * static_cast<double>(sizes.size() - 1)));
        return sizes[index];
    };
    const auto mean = std::accumulate(
        sizes.begin(), sizes.end(), static_cast<long double>(0.0L)) /
        static_cast<long double>(sizes.size());
    std::cout << "ml_contraction_metrics level=" << level
              << " vertex_ratio=" << static_cast<double>(coarse.vertices()) /
                     static_cast<double>(fine.vertices())
              << " edge_ratio=" << static_cast<double>(coarse.edges()) /
                     static_cast<double>(fine.edges())
              << " internal_edge_ratio=" <<
                     (total_edge_weight == 0 ? 0.0 :
                      static_cast<double>(internal_edge_weight) /
                      static_cast<double>(total_edge_weight))
              << " aggregate_mean=" << static_cast<double>(mean)
              << " aggregate_p50=" << percentile(0.50)
              << " aggregate_p90=" << percentile(0.90)
              << " aggregate_p99=" << percentile(0.99)
              << " aggregate_max=" << (sizes.empty() ? 0 : sizes.back())
              << '\n';
}

void validate_weighted_csr(const WeightedGraph& graph, bool allow_self_loops) {
    if (graph.offsets.size() != graph.vertex_weights.size() + 1 ||
        graph.offsets.empty() || graph.offsets.front() != 0 ||
        graph.offsets.back() != static_cast<std::int64_t>(graph.neighbors.size()) ||
        graph.neighbors.size() != graph.edge_weights.size()) {
        throw std::runtime_error("invalid weighted CSR shape");
    }
    for (std::int64_t v = 0; v < graph.vertices(); ++v) {
        const auto begin = graph.offsets[static_cast<std::size_t>(v)];
        const auto end = graph.offsets[static_cast<std::size_t>(v + 1)];
        if (begin > end) throw std::runtime_error("weighted CSR offsets are not monotone");
        for (auto e = begin; e < end; ++e) {
            const auto u = graph.neighbors[static_cast<std::size_t>(e)];
            if (u < 0 || u >= graph.vertices()) {
                throw std::runtime_error("weighted CSR contains an invalid neighbor");
            }
            if (!allow_self_loops && u == v) {
                throw std::runtime_error("contracted graph contains a self loop");
            }
            const auto reverse_begin = graph.offsets[static_cast<std::size_t>(u)];
            const auto reverse_end = graph.offsets[static_cast<std::size_t>(u + 1)];
            const auto reverse = std::lower_bound(
                graph.neighbors.begin() + reverse_begin,
                graph.neighbors.begin() + reverse_end,
                static_cast<std::int32_t>(v));
            if (reverse == graph.neighbors.begin() + reverse_end || *reverse != v) {
                throw std::runtime_error("weighted CSR is not bidirectional");
            }
            const auto reverse_edge = static_cast<std::size_t>(
                reverse - graph.neighbors.begin());
            if (graph.edge_weights[static_cast<std::size_t>(e)] !=
                graph.edge_weights[reverse_edge]) {
                throw std::runtime_error("weighted CSR reverse edge weights differ");
            }
        }
    }
}

void validate_weighted_shape(const WeightedGraph& graph, bool allow_self_loops) {
    if (graph.offsets.size() != graph.vertex_weights.size() + 1 ||
        graph.offsets.empty() || graph.offsets.front() != 0 ||
        graph.offsets.back() != static_cast<std::int64_t>(graph.neighbors.size()) ||
        graph.neighbors.size() != graph.edge_weights.size()) {
        throw std::runtime_error("invalid weighted CSR shape");
    }
    for (std::int64_t v = 0; v < graph.vertices(); ++v) {
        const auto begin = graph.offsets[static_cast<std::size_t>(v)];
        const auto end = graph.offsets[static_cast<std::size_t>(v + 1)];
        if (begin > end) throw std::runtime_error("weighted CSR offsets are not monotone");
        for (auto e = begin; e < end; ++e) {
            const auto u = graph.neighbors[static_cast<std::size_t>(e)];
            if (u < 0 || u >= graph.vertices()) {
                throw std::runtime_error("weighted CSR contains an invalid neighbor");
            }
            if (!allow_self_loops && u == v) {
                throw std::runtime_error("contracted graph contains a self loop");
            }
        }
    }
}

void validate_coarsening_step(
    const WeightedGraph& fine, const WeightedGraph& coarse,
    const std::vector<std::int32_t>& map, std::uint64_t cluster_cap,
    int level, bool strict_verify) {
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
        recomputed[static_cast<std::size_t>(c)] +=
            fine.vertex_weights[static_cast<std::size_t>(v)];
        ++coverage[static_cast<std::size_t>(c)];
    }
    const auto fine_total = std::accumulate(
        fine.vertex_weights.begin(), fine.vertex_weights.end(), std::uint64_t{0});
    const auto coarse_total = std::accumulate(
        coarse.vertex_weights.begin(), coarse.vertex_weights.end(), std::uint64_t{0});
    if (fine_total != coarse_total) {
        throw std::runtime_error("coarsening did not conserve vertex weight");
    }
    for (std::int64_t c = 0; c < coarse.vertices(); ++c) {
        if (coverage[static_cast<std::size_t>(c)] == 0 ||
            recomputed[static_cast<std::size_t>(c)] !=
                coarse.vertex_weights[static_cast<std::size_t>(c)]) {
            throw std::runtime_error("coarsening cluster weight mismatch");
        }
        if (coarse.vertex_weights[static_cast<std::size_t>(c)] > cluster_cap) {
            throw std::runtime_error("coarse point exceeds configured weight cap");
        }
    }
    if (strict_verify) {
        validate_weighted_csr(coarse, false);
    } else {
        validate_weighted_shape(coarse, false);
    }

    // A random coarse labeling must have exactly the same weighted cut after
    // projection.  This checks both map direction and parallel-edge merging.
    std::vector<std::int32_t> coarse_labels(static_cast<std::size_t>(coarse.vertices()));
    for (std::int64_t c = 0; c < coarse.vertices(); ++c) {
        coarse_labels[static_cast<std::size_t>(c)] =
            static_cast<std::int32_t>((c * 2654435761ULL + 17) % 23);
    }
    std::vector<std::int32_t> fine_labels(map.size());
    for (std::size_t v = 0; v < map.size(); ++v) {
        fine_labels[v] = coarse_labels[static_cast<std::size_t>(map[v])];
    }
    if (host_cut(fine, fine_labels) != host_cut(coarse, coarse_labels)) {
        throw std::runtime_error("cut is not conserved by coarse projection");
    }
    std::cout << "ml_layer_verify level=" << level
              << " map=ok weights=ok csr=ok projection_cut=ok\n";
}

void write_jet_hierarchy(
    const std::vector<WeightedGraph>& levels,
    const std::vector<std::vector<std::int32_t>>& maps,
    const std::string& path) {
    if (levels.empty() || maps.size() + 1 != levels.size()) {
        throw std::runtime_error("cannot write an incomplete Jet hierarchy");
    }
    const auto temporary = path + ".tmp";
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open hierarchy temporary file");
        const auto level_count = static_cast<std::int32_t>(levels.size());
        output.write(reinterpret_cast<const char*>(&level_count), sizeof(level_count));
        for (std::size_t i = 0; i < levels.size(); ++i) {
            const auto& graph = levels[i];
            if (graph.vertices() > std::numeric_limits<std::int32_t>::max() ||
                graph.edges() > std::numeric_limits<std::int32_t>::max()) {
                throw std::runtime_error("Jet standard hierarchy requires 32-bit dimensions");
            }
            const auto n = static_cast<std::int32_t>(graph.vertices());
            const auto m = static_cast<std::int32_t>(graph.edges());
            output.write(reinterpret_cast<const char*>(&n), sizeof(n));
            output.write(reinterpret_cast<const char*>(&m), sizeof(m));
            for (const auto offset : graph.offsets) {
                if (offset > std::numeric_limits<std::int32_t>::max()) {
                    throw std::runtime_error("Jet row offset exceeds int32");
                }
                const auto value = static_cast<std::int32_t>(offset);
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            for (const auto neighbor : graph.neighbors) {
                output.write(reinterpret_cast<const char*>(&neighbor), sizeof(neighbor));
            }
            for (const auto weight : graph.edge_weights) {
                if (weight > std::numeric_limits<std::int32_t>::max()) {
                    throw std::runtime_error("Jet edge weight exceeds int32");
                }
                const auto value = static_cast<std::int32_t>(weight);
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            for (const auto weight : graph.vertex_weights) {
                if (weight > std::numeric_limits<std::int32_t>::max()) {
                    throw std::runtime_error("Jet vertex weight exceeds int32");
                }
                const auto value = static_cast<std::int32_t>(weight);
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            if (i > 0) {
                const auto& map = maps[i - 1];
                if (map.size() != static_cast<std::size_t>(levels[i - 1].vertices())) {
                    throw std::runtime_error("Jet map has the wrong fine-level length");
                }
                for (const auto coarse : map) {
                    if (coarse < 0 || coarse >= n) {
                        throw std::runtime_error("Jet map contains an invalid coarse id");
                    }
                    output.write(reinterpret_cast<const char*>(&coarse), sizeof(coarse));
                }
            }
        }
        output.flush();
        if (!output) throw std::runtime_error("failed while writing Jet hierarchy");
        output.close();
        if (std::rename(temporary.c_str(), path.c_str()) != 0) {
            throw std::runtime_error("cannot commit Jet hierarchy output");
        }
    } catch (...) {
        std::remove(temporary.c_str());
        throw;
    }
}

WeightedGraph make_weighted(const CSRGraph& graph) {
    WeightedGraph out;
    out.offsets = graph.offsets();
    out.neighbors = graph.neighbors();
    out.edge_weights.assign(static_cast<std::size_t>(graph.edges()), 1);
    out.vertex_weights.assign(static_cast<std::size_t>(graph.vertices()), 1);
    return out;
}

void run_device_hierarchy(
    const CSRGraph& input, int parts, double ratio, std::uint32_t seed,
    double stop_contraction_ratio, int basc_k, int max_levels,
    const std::string& method, const std::string& output_path,
    bool strict_verify) {
    const auto total_start = std::chrono::steady_clock::now();
    const bool frontier_method = method == "frontier";
    const bool sclp_method = method == "sclp";
    const bool diagnostics = sclp_method
        ? std::getenv("SCLP_DIAGNOSTICS") != nullptr
        : (frontier_method
               ? std::getenv("FRONTIER_DIAGNOSTICS") != nullptr
               : std::getenv("BASC_DIAGNOSTICS") != nullptr);
    const bool verify = strict_verify ||
        (sclp_method
             ? std::getenv("SCLP_VERIFY") != nullptr
             : (frontier_method
                    ? std::getenv("FRONTIER_VERIFY") != nullptr
                    : std::getenv("BASC_GPU_VERIFY") != nullptr));
    const bool skip_export = std::getenv("ML_SKIP_HIERARCHY_EXPORT") != nullptr;
    std::vector<WeightedGraph> levels;
    std::vector<std::vector<std::int32_t>> maps;
    levels.push_back(make_weighted(input));

    const auto input_verify_start = std::chrono::steady_clock::now();
    if (strict_verify) {
        validate_weighted_csr(levels.front(), true);
    } else {
        validate_weighted_shape(levels.front(), true);
        std::vector<std::int32_t> labels(
            static_cast<std::size_t>(levels.front().vertices()));
        for (std::int64_t v = 0; v < levels.front().vertices(); ++v) {
            labels[static_cast<std::size_t>(v)] =
                static_cast<std::int32_t>((v * 2654435761ULL + 17) % 23);
        }
        (void)host_cut(levels.front(), labels);
    }
    std::cout << "ml_input_verify_seconds="
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - input_verify_start).count()
              << " status=ok mode=" << (strict_verify ? "strict" : "fast") << '\n';

    const auto device_input_start = std::chrono::steady_clock::now();
    DeviceWeightedGraph current = make_device_weighted(levels.front());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto device_input_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - device_input_start).count();
    std::cout << "ml_gpu_device_input_seconds=" << device_input_seconds
              << " method=" << method << '\n';

    const std::int64_t cutoff = std::max<std::int64_t>(32, parts * 8);
    const auto core_start = std::chrono::steady_clock::now();
    double snapshot_seconds = 0.0;
    std::string stop_reason = "vertex_cutoff";
    int level = 0;
    for (; level < max_levels && current.vertices() > cutoff; ++level) {
        std::cout << "ml_coarsen_begin level=" << level
                  << " vertices=" << current.vertices()
                  << " edges=" << current.edges()
                  << " method=" << method
                  << " seed=" << (seed + static_cast<std::uint32_t>(level)) << '\n';
        DeviceAggregateResult aggregate;
        if (sclp_method) {
            SclpStats stats;
            aggregate = gpu_sclp_aggregate_device(
                current, parts, seed + static_cast<std::uint32_t>(level),
                level, stats, diagnostics);
        } else if (frontier_method) {
            FrontierStats stats;
            aggregate = gpu_frontier_aggregate_device(
                current, parts, seed + static_cast<std::uint32_t>(level),
                level, stats, diagnostics);
        } else {
            BascStats stats;
            aggregate = gpu_basc_aggregate_device(
                current, parts, basc_k,
                seed + static_cast<std::uint32_t>(level), level, stats, diagnostics);
        }
        const double contraction = static_cast<double>(aggregate.coarse_vertices) /
                                   static_cast<double>(current.vertices());
        std::cout << "ml_coarsen_map level=" << level
                  << " coarse_vertices=" << aggregate.coarse_vertices
                  << " ratio=" << contraction
                  << " maximum_weight=" << aggregate.maximum_weight
                  << " capacity=" << aggregate.capacity << '\n';
        if (aggregate.coarse_vertices >= current.vertices() ||
            contraction > stop_contraction_ratio) {
            std::cout << "ml_coarsen_stop reason=insufficient_contraction\n";
            stop_reason = "insufficient_contraction";
            break;
        }

        double contract_seconds = 0.0;
        auto coarse_device = gpu_contract_graph(current, aggregate, contract_seconds);
        std::cout << "ml_gpu_contract_seconds level=" << level
                  << " seconds=" << contract_seconds
                  << " coarse_edges=" << coarse_device.edges() << '\n';

        const auto snapshot_start = std::chrono::steady_clock::now();
        std::vector<std::int32_t> host_map(aggregate.map.size());
        thrust::copy(aggregate.map.begin(), aggregate.map.end(), host_map.begin());
        auto coarse_host = copy_device_weighted(coarse_device);
        const auto snapshot = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - snapshot_start).count();
        snapshot_seconds += snapshot;
        std::cout << "ml_gpu_device_snapshot_seconds level=" << level
                  << " seconds=" << snapshot << '\n';

        if (verify) {
            const auto verify_start = std::chrono::steady_clock::now();
            validate_coarsening_step(
                levels.back(), coarse_host, host_map, aggregate.capacity,
                level, strict_verify);
            std::cout << "ml_gpu_device_verify_seconds level=" << level
                      << " seconds=" << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - verify_start).count()
                      << '\n';
        }
        maps.push_back(std::move(host_map));
        levels.push_back(std::move(coarse_host));
        current = std::move(coarse_device);
    }
    if (level == max_levels && current.vertices() > cutoff) {
        stop_reason = "max_levels";
    }
    const auto core_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - core_start).count();
    std::cout << "ml_gpu_device_core_seconds=" << core_seconds
              << " aggregate_plus_contract=1 method=" << method << '\n';

    const auto export_start = std::chrono::steady_clock::now();
    if (!skip_export) write_jet_hierarchy(levels, maps, output_path);
    const auto export_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - export_start).count();
    std::cout << "ml_hierarchy_export_seconds=" << export_seconds
              << " skipped=" << (skip_export ? 1 : 0) << '\n';
    std::cout << "ml_hierarchy_levels=" << levels.size()
              << " coarsest_vertices=" << levels.back().vertices()
              << " stop_reason=" << stop_reason
              << " stop_contraction_ratio=" << stop_contraction_ratio
              << " method=" << method
              << " basc_k=" << basc_k << '\n';
    std::cout << "ml_gpu_device_snapshot_total_seconds=" << snapshot_seconds
              << " method=" << method << '\n';
    std::cout << "ml_total_seconds="
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - total_start).count()
              << " hierarchy=" << (skip_export ? "skipped" : output_path)
              << " coarsen_only=1 method=" << method
              << " basc_k=" << basc_k << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 11) {
            std::cerr << "Usage: " << argv[0]
                      << " <indptr.bin> <indices.bin> <parts> <hierarchy.out>"
                      << " [max_vertex_ratio] [seed] [stop_contraction_ratio]"
                      << " [coarsen_method=lp|basc|basc_gpu|frontier|sclp] [basc_k=1|2|4]"
                      << " [max_levels]\n";
            return 2;
        }
        const int parts = std::stoi(argv[3]);
        const double ratio = argc >= 6 ? std::stod(argv[5]) : 1.10;
        const std::uint32_t seed = argc >= 7
            ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 0U;
        const double stop_contraction_ratio = argc >= 8
            ? std::stod(argv[7]) : 0.90;
        const std::string coarsen_method = argc >= 9 ? argv[8] : "lp";
        const int basc_k = argc >= 10 ? std::stoi(argv[9]) : 2;
        const int max_levels = argc >= 11 ? std::stoi(argv[10]) : 24;
        if (parts < 2 || parts > 32 || ratio < 1.0) {
            throw std::runtime_error("invalid parts or maximum vertex ratio");
        }
        if (coarsen_method != "lp" && coarsen_method != "basc" &&
            coarsen_method != "basc_gpu" && coarsen_method != "frontier" &&
            coarsen_method != "sclp") {
            throw std::runtime_error(
                "coarsen_method must be lp, basc, basc_gpu, frontier, or sclp");
        }
        if (stop_contraction_ratio <= 0.0 || stop_contraction_ratio > 1.0) {
            throw std::runtime_error("invalid coarsening stop ratio");
        }
        if (max_levels <= 0 || max_levels > 200) {
            throw std::runtime_error("invalid coarsening max levels");
        }
        constexpr int aggregate_rounds = 2;
        const std::int64_t cutoff = std::max<std::int64_t>(32, parts * 8);

        const auto total_start = std::chrono::steady_clock::now();
        CSRGraph input;
        input.load(argv[1], argv[2]);
        const bool strict_verify = std::getenv("GPU_LP_STRICT_VERIFY") != nullptr;
        if (coarsen_method == "basc_gpu" || coarsen_method == "frontier" ||
            coarsen_method == "sclp") {
            run_device_hierarchy(
                input, parts, ratio, seed, stop_contraction_ratio,
                basc_k, max_levels, coarsen_method, argv[4], strict_verify);
            return 0;
        }
        std::vector<WeightedGraph> levels;
        std::vector<std::vector<std::int32_t>> maps;
        levels.push_back(make_weighted(input));
        const auto input_verify_start = std::chrono::steady_clock::now();
        if (strict_verify) {
            validate_weighted_csr(levels.front(), true);
        } else {
            validate_weighted_shape(levels.front(), true);
            std::vector<std::int32_t> labels(
                static_cast<std::size_t>(levels.front().vertices()));
            for (std::int64_t v = 0; v < levels.front().vertices(); ++v) {
                labels[static_cast<std::size_t>(v)] =
                    static_cast<std::int32_t>((v * 2654435761ULL + 17) % 23);
            }
            // The input convention is symmetric CSR.  This linear check
            // catches odd directed cut totals without adding a second full
            // adjacency index to a multi-million-vertex input.
            (void)host_cut(levels.front(), labels);
        }
        std::cout << "ml_input_verify_seconds="
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - input_verify_start).count()
                  << " status=ok mode=" << (strict_verify ? "strict" : "fast") << '\n';

        std::string stop_reason = "vertex_cutoff";
        int level = 0;
        for (; level < max_levels && levels.back().vertices() > cutoff; ++level) {
            const auto& fine = levels.back();
            const auto total_weight = std::accumulate(
                fine.vertex_weights.begin(), fine.vertex_weights.end(), std::uint64_t{0});
            const auto maximum_vertex = *std::max_element(
                fine.vertex_weights.begin(), fine.vertex_weights.end());
            const auto target_vertices = std::max<std::int64_t>(cutoff, fine.vertices() / 2);
            const auto average_target_weight =
                (total_weight + static_cast<std::uint64_t>(target_vertices) - 1) /
                static_cast<std::uint64_t>(target_vertices);
            const auto partition_capacity = static_cast<std::uint64_t>(std::floor(
                static_cast<long double>(total_weight) * ratio / parts));
            const auto cluster_cap = std::min(
                partition_capacity,
                std::max(maximum_vertex, average_target_weight * 2));
            std::cout << "ml_coarsen_begin level=" << level
                      << " vertices=" << fine.vertices()
                      << " edges=" << fine.edges()
                      << " lp_cluster_cap=" << cluster_cap
                      << " method=" << coarsen_method
                      << " seed=" << (seed + static_cast<std::uint32_t>(level)) << '\n';
            AggregateResult aggregate;
            if (coarsen_method == "basc") {
                BascStats basc_stats;
                aggregate = gpu_basc_aggregate(
                    fine, parts, basc_k,
                    seed + static_cast<std::uint32_t>(level), level, basc_stats);
            } else {
                aggregate = gpu_size_constrained_lp_aggregate(
                    fine, aggregate_rounds, cluster_cap,
                    seed + static_cast<std::uint32_t>(level));
            }
            const double contraction = static_cast<double>(aggregate.coarse_vertices) /
                                       static_cast<double>(fine.vertices());
            std::cout << "ml_coarsen_map level=" << level
                      << " coarse_vertices=" << aggregate.coarse_vertices
                      << " ratio=" << contraction
                      << " maximum_weight=" << aggregate.maximum_weight
                      << " capacity=" << aggregate.capacity << '\n';
            if (aggregate.coarse_vertices >= fine.vertices() ||
                contraction > stop_contraction_ratio) {
                std::cout << "ml_coarsen_stop reason=insufficient_contraction\n";
                stop_reason = "insufficient_contraction";
                break;
            }
            const auto contract_start = std::chrono::steady_clock::now();
            auto coarse = contract_graph(fine, aggregate);
            const auto contract_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - contract_start).count();
            std::cout << "ml_cpu_contract_seconds level=" << level
                      << " seconds=" << contract_seconds << '\n';
            report_contraction_metrics(fine, coarse, aggregate.map, level);
            validate_coarsening_step(
                fine, coarse, aggregate.map, aggregate.capacity,
                level, strict_verify);
            maps.push_back(std::move(aggregate.map));
            levels.push_back(std::move(coarse));
        }
        if (level == max_levels && levels.back().vertices() > cutoff) {
            stop_reason = "max_levels";
        }
        write_jet_hierarchy(levels, maps, argv[4]);
        std::cout << "ml_hierarchy_levels=" << levels.size()
                  << " coarsest_vertices=" << levels.back().vertices()
                  << " stop_reason=" << stop_reason
                  << " stop_contraction_ratio=" << stop_contraction_ratio
                  << " method=" << coarsen_method
                  << " basc_k=" << basc_k << '\n';
        std::cout << "ml_total_seconds="
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - total_start).count()
                  << " hierarchy=" << argv[4]
                  << " coarsen_only=1 method=" << coarsen_method
                  << " basc_k=" << basc_k << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multilevel_lp: " << error.what() << '\n';
        return 1;
    }
}
