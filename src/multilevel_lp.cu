#include "check.hpp"
#include "graph.hpp"

#include <algorithm>
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
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
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

struct BascStats {
    std::uint64_t anchors = 0;
    std::uint64_t accepted_first_round = 0;
    std::uint64_t accepted_expansion = 0;
    std::uint64_t singleton_count = 0;
    std::uint64_t cluster_cap = 0;
    double anchor_seconds = 0.0;
    double candidate_seconds = 0.0;
    double support_seconds = 0.0;
    double admission_seconds = 0.0;
    double expansion_seconds = 0.0;
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
    std::int32_t* candidates, float* affinities) {
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
        return;
    }
    for (auto e = offsets[v]; e < offsets[v + 1]; ++e) {
        const auto u = neighbors[e];
        if (!anchors[u]) continue;
        const float denominator = sqrtf(static_cast<float>(degrees[u]) + 1.0f);
        const float affinity = static_cast<float>(edge_weights[e]) / denominator;
        basc_insert_candidate(k, u, affinity, my_candidates, my_affinities);
    }
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
    unsigned long long* aggregate_weights, unsigned long long* accepted) {
    const auto v = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (v >= n || buckets[v] != bucket || aggregates[v] != BASC_INVALID) return;
    const auto target = proposals[v];
    if (target < 0) return;
    const auto weight = static_cast<unsigned long long>(vertex_weights[v]);
    auto* slot = &aggregate_weights[target];
    auto old = atomicAdd(slot, 0ULL);
    while (old + weight <= capacity) {
        const auto previous = atomicCAS(slot, old, old + weight);
        if (previous == old) {
            aggregates[v] = target;
            atomicAdd(accepted, 1ULL);
            return;
        }
        old = previous;
    }
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
    unsigned long long* accepted) {
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
    const auto weight = static_cast<unsigned long long>(vertex_weights[v]);
    auto* slot = &aggregate_weights[first];
    auto old = atomicAdd(slot, 0ULL);
    while (old + weight <= capacity) {
        const auto previous = atomicCAS(slot, old, old + weight);
        if (previous == old) {
            aggregates[v] = first;
            atomicAdd(accepted, 1ULL);
            return;
        }
        old = previous;
    }
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
    thrust::device_vector<std::int32_t> proposals(static_cast<std::size_t>(n));
    thrust::device_vector<float> confidence(static_cast<std::size_t>(n));
    thrust::device_vector<std::uint8_t> buckets(static_cast<std::size_t>(n));
    thrust::device_vector<std::int32_t> aggregates(static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> aggregate_weights(
        static_cast<std::size_t>(n));
    thrust::device_vector<unsigned long long> counter(1);
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
        thrust::raw_pointer_cast(affinities.data()));
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
    CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counter.data()), 0,
                          sizeof(unsigned long long)));
    const auto admission_start = std::chrono::steady_clock::now();
    for (int bucket = 3; bucket >= 0; --bucket) {
        basc_admit_bucket_kernel<<<blocks, 256>>>(
            n, bucket, stats.cluster_cap,
            thrust::raw_pointer_cast(vertex_weights.data()),
            thrust::raw_pointer_cast(proposals.data()),
            thrust::raw_pointer_cast(buckets.data()),
            thrust::raw_pointer_cast(aggregates.data()),
            thrust::raw_pointer_cast(aggregate_weights.data()),
            thrust::raw_pointer_cast(counter.data()));
        CUDA_CHECK(cudaGetLastError());
    }
    unsigned long long accepted_first_round = 0;
    CUDA_CHECK(cudaMemcpy(&accepted_first_round,
        thrust::raw_pointer_cast(counter.data()), sizeof(accepted_first_round),
        cudaMemcpyDeviceToHost));
    stats.accepted_first_round = accepted_first_round;
    stats.admission_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - admission_start).count();

    CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counter.data()), 0,
                          sizeof(unsigned long long)));
    const auto expansion_start = std::chrono::steady_clock::now();
    basc_expansion_kernel<<<blocks, 256>>>(
        n, 0.25f, stats.cluster_cap,
        thrust::raw_pointer_cast(offsets.data()),
        thrust::raw_pointer_cast(neighbors.data()),
        thrust::raw_pointer_cast(edge_weights.data()),
        thrust::raw_pointer_cast(vertex_weights.data()),
        thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(aggregate_weights.data()),
        thrust::raw_pointer_cast(counter.data()));
    CUDA_CHECK(cudaGetLastError());
    unsigned long long accepted_expansion = 0;
    CUDA_CHECK(cudaMemcpy(&accepted_expansion,
        thrust::raw_pointer_cast(counter.data()), sizeof(accepted_expansion),
        cudaMemcpyDeviceToHost));
    stats.accepted_expansion = accepted_expansion;
    stats.expansion_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - expansion_start).count();

    CUDA_CHECK(cudaMemset(thrust::raw_pointer_cast(counter.data()), 0,
                          sizeof(unsigned long long)));
    basc_singleton_kernel<<<blocks, 256>>>(
        n, thrust::raw_pointer_cast(vertex_weights.data()),
        thrust::raw_pointer_cast(aggregates.data()),
        thrust::raw_pointer_cast(aggregate_weights.data()),
        thrust::raw_pointer_cast(counter.data()));
    CUDA_CHECK(cudaGetLastError());
    unsigned long long singleton_count = 0;
    CUDA_CHECK(cudaMemcpy(&singleton_count,
        thrust::raw_pointer_cast(counter.data()), sizeof(singleton_count),
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
              << " accepted_first_round=" << stats.accepted_first_round
              << " accepted_expansion=" << stats.accepted_expansion
              << " singletons=" << stats.singleton_count
              << " cluster_cap=" << stats.cluster_cap
              << " anchor_seconds=" << stats.anchor_seconds
              << " candidate_seconds=" << stats.candidate_seconds
              << " support_seconds=" << stats.support_seconds
              << " admission_seconds=" << stats.admission_seconds
              << " expansion_seconds=" << stats.expansion_seconds << '\n';
    return out;
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

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 11) {
            std::cerr << "Usage: " << argv[0]
                      << " <indptr.bin> <indices.bin> <parts> <hierarchy.out>"
                      << " [max_vertex_ratio] [seed] [stop_contraction_ratio]"
                      << " [coarsen_method=lp|basc] [basc_k=1|2|4]"
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
        if (coarsen_method != "lp" && coarsen_method != "basc") {
            throw std::runtime_error("coarsen_method must be lp or basc");
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
        std::vector<WeightedGraph> levels;
        std::vector<std::vector<std::int32_t>> maps;
        levels.push_back(make_weighted(input));
        const bool strict_verify = std::getenv("GPU_LP_STRICT_VERIFY") != nullptr;
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
