#include "partitioner.hpp"
#include "check.hpp"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

constexpr int kGainBins = 4096;
constexpr std::int64_t kFieldGainBound = 1000000;
constexpr std::int64_t kBalanceGainBound = 1000000;
constexpr std::int32_t kBfsInf = 0x3f3f3f3f;

__host__ __device__ __forceinline__ std::uint32_t hash32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

struct MovePredicate {
    const std::int32_t* targets;
    const std::int32_t* gains;
    const std::int32_t* labels;
    const unsigned long long* loads;
    std::int64_t capacity;
    int min_gain;
    bool repair;

    __device__ bool operator()(const std::int64_t& vertex) const {
        const int source = labels[vertex];
        const int target = targets[vertex];
        if (target == source || target < 0) return false;
        if (loads[target] >= static_cast<std::uint64_t>(capacity)) return false;
        return gains[vertex] >= min_gain ||
               (repair && loads[source] > static_cast<std::uint64_t>(capacity));
    }
};

struct DegreePredicate {
    const std::int64_t* offsets;
    int threshold;

    __device__ bool operator()(const std::int64_t& vertex) const {
        return offsets[vertex + 1] - offsets[vertex] >= threshold;
    }
};

struct MovedPredicate {
    const std::uint8_t* moved_flags;

    __device__ bool operator()(const std::int64_t& vertex) const {
        return moved_flags[vertex] != 0;
    }
};


struct PairCandidatePredicate {
    const std::int32_t* targets;
    int top_targets;

    __device__ bool operator()(const std::int64_t& value) const {
        const auto vertex = value / top_targets;
        const auto slot = value - vertex * top_targets;
        return targets[vertex * top_targets + slot] >= 0;
    }
};

__device__ __forceinline__ void block_vertex_gain(
    std::int32_t vertex, int source, int target, int parts,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::int32_t* neighbor_label_counts, const std::int32_t* labels,
    std::int64_t& target_count, std::int64_t& source_count) {
    target_count = 0;
    source_count = 0;
    if (neighbor_label_counts) {
        const auto base = static_cast<std::size_t>(vertex) * parts;
        target_count = neighbor_label_counts[base + target];
        source_count = neighbor_label_counts[base + source];
        for (std::int64_t edge = offsets[vertex];
             edge < offsets[vertex + 1]; ++edge) {
            if (neighbors[edge] == vertex) --source_count;
        }
        return;
    }
    for (std::int64_t edge = offsets[vertex];
         edge < offsets[vertex + 1]; ++edge) {
        const auto neighbor = neighbors[edge];
        if (neighbor == vertex) continue;
        const int label = labels[neighbor];
        target_count += label == target;
        source_count += label == source;
    }
}

__global__ void fill_index_kernel(
    std::int64_t count, std::int64_t* values) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index < count) values[index] = index;
}

__global__ void block_seed_sort_key_kernel(
    std::int64_t count, int target, int parts, int stage,
    const std::int64_t* values, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::int32_t* neighbor_label_counts,
    const std::int32_t* labels, std::uint64_t* keys) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= count) return;
    const auto vertex = static_cast<std::int32_t>(values[index]);
    const int source = labels[vertex];
    std::int64_t target_count = 0;
    std::int64_t source_count = 0;
    block_vertex_gain(vertex, source, target, parts, offsets, neighbors,
                      neighbor_label_counts, labels, target_count, source_count);
    const bool valid = source >= 0 && source < parts && source != target &&
                       target_count > 0;
    if (stage == 0) {
        keys[index] = static_cast<std::uint64_t>(vertex);
    } else if (stage == 1) {
        const auto raw_gain = target_count - source_count;
        const auto gain = valid
            ? static_cast<std::int32_t>(
                raw_gain > 2147483647LL ? 2147483647LL
                : (raw_gain < -2147483647LL - 1LL
                       ? -2147483647LL - 1LL : raw_gain))
            : static_cast<std::int32_t>(-2147483647LL - 1LL);
        const auto sortable = static_cast<std::uint32_t>(gain) ^ 0x80000000U;
        keys[index] = valid ? ~static_cast<std::uint64_t>(sortable)
                            : ~static_cast<std::uint64_t>(0);
    } else {
        keys[index] = valid ? static_cast<std::uint64_t>(source)
                            : static_cast<std::uint64_t>(parts);
    }
}

__global__ void count_block_seed_buckets_kernel(
    std::int64_t count, int target, int parts,
    const std::int64_t* values, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::int32_t* neighbor_label_counts,
    const std::int32_t* labels, std::int32_t* bucket_counts) {
    const auto index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= count) return;
    const auto vertex = static_cast<std::int32_t>(values[index]);
    const int source = labels[vertex];
    std::int64_t target_count = 0;
    std::int64_t source_count = 0;
    block_vertex_gain(vertex, source, target, parts, offsets, neighbors,
                      neighbor_label_counts, labels, target_count, source_count);
    if (source >= 0 && source < parts && source != target && target_count > 0) {
        atomicAdd(&bucket_counts[source * parts + target], 1);
    }
}

__global__ void extract_block_seeds_kernel(
    int parts, int seeds_per_pair, int target,
    const std::int64_t* sorted_values, const std::int32_t* bucket_counts,
    const std::int32_t* bucket_begin, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::int32_t* neighbor_label_counts,
    const std::int32_t* labels, std::int32_t* seed_vertices,
    std::int64_t* seed_gains, std::int32_t* seed_sources,
    std::int32_t* seed_targets) {
    const int item = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = parts * seeds_per_pair;
    if (item >= total) return;
    const int source = item / seeds_per_pair;
    const int slot = item - source * seeds_per_pair;
    if (source == target) return;
    const int bucket = source * parts + target;
    if (slot >= bucket_counts[bucket]) return;
    const auto value = sorted_values[bucket_begin[bucket] + slot];
    const auto vertex = static_cast<std::int32_t>(value);
    const int pair_index = source * (parts - 1) +
                           (target < source ? target : target - 1);
    const int output = pair_index * seeds_per_pair + slot;
    std::int64_t target_count = 0;
    std::int64_t source_count = 0;
    block_vertex_gain(vertex, source, target, parts, offsets, neighbors,
                      neighbor_label_counts, labels, target_count, source_count);
    seed_vertices[output] = vertex;
    seed_gains[output] = target_count - source_count;
    seed_sources[output] = source;
    seed_targets[output] = target;
}

__device__ __forceinline__ bool block_contains_vertex(
    const std::int32_t* vertices, int size, std::int32_t vertex) {
    for (int index = 0; index < size; ++index) {
        if (vertices[index] == vertex) return true;
    }
    return false;
}

__global__ void block_growth_kernel(
    int candidate_count, int max_size, int frontier_limit,
    const std::int32_t* seed_vertices, const std::int64_t* seed_gains,
    const std::int32_t* seed_sources, const std::int32_t* seed_targets,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::int32_t* labels, std::int32_t* block_vertices,
    std::int64_t* candidate_gains, std::int32_t* candidate_sizes,
    std::uint8_t* candidate_valid, std::int32_t* frontier_vertices,
    std::int64_t* frontier_scores, std::uint64_t* frontier_overflows,
    std::uint64_t* edge_visits) {
    const int candidate = blockIdx.x;
    if (candidate >= candidate_count) return;
    const int seed = seed_vertices[candidate];
    if (seed < 0) return;
    const int source = seed_sources[candidate];
    const int target = seed_targets[candidate];
    const auto vertex_base = static_cast<std::size_t>(candidate) * max_size;
    const auto frontier_base = static_cast<std::size_t>(candidate) *
                               frontier_limit;
    __shared__ int current_size;
    __shared__ int frontier_count;
    __shared__ int frontier_lock;
    __shared__ int frontier_overflow;
    __shared__ int overflow_rounds;
    __shared__ int next_vertex;
    __shared__ std::int64_t next_delta;
    __shared__ std::int64_t current_gain;
    __shared__ std::int64_t best_gain;
    __shared__ int best_size;
    if (threadIdx.x == 0) {
        current_size = 1;
        current_gain = seed_gains[candidate];
        best_gain = current_gain > 0 ? current_gain : 0;
        best_size = current_gain > 0 ? 1 : 0;
        block_vertices[vertex_base] = seed;
        overflow_rounds = 0;
    }
    __syncthreads();

    std::uint64_t local_visits = 0;
    for (int step = 1; step < max_size; ++step) {
        if (threadIdx.x == 0) {
            frontier_count = 0;
            frontier_lock = 0;
            frontier_overflow = 0;
            next_vertex = -1;
            next_delta = -9223372036854775807LL - 1LL;
        }
        __syncthreads();
        for (int member = threadIdx.x; member < current_size;
             member += blockDim.x) {
            const auto member_vertex = block_vertices[vertex_base + member];
            for (std::int64_t edge = offsets[member_vertex];
                 edge < offsets[member_vertex + 1]; ++edge) {
                ++local_visits;
                const auto neighbor = neighbors[edge];
                if (labels[neighbor] != source ||
                    block_contains_vertex(&block_vertices[vertex_base],
                                           current_size, neighbor)) {
                    continue;
                }
                while (atomicCAS(&frontier_lock, 0, 1) != 0) {}
                int count = frontier_count;
                bool present = false;
                for (int index = 0; index < count; ++index) {
                    if (frontier_vertices[frontier_base + index] == neighbor) {
                        present = true;
                        break;
                    }
                }
                if (!present) {
                    const auto hash = hash32(
                        static_cast<std::uint32_t>(seed) ^
                        static_cast<std::uint32_t>(target * 131) ^
                        static_cast<std::uint32_t>(neighbor));
                    if (count < frontier_limit) {
                        frontier_vertices[frontier_base + count] = neighbor;
                        ++frontier_count;
                    } else {
                        frontier_overflow = 1;
                        int worst = 0;
                        auto worst_hash = hash32(
                            static_cast<std::uint32_t>(seed) ^
                            static_cast<std::uint32_t>(target * 131) ^
                            static_cast<std::uint32_t>(
                                frontier_vertices[frontier_base]));
                        for (int index = 1; index < frontier_limit; ++index) {
                            const auto existing = frontier_vertices[
                                frontier_base + index];
                            const auto existing_hash = hash32(
                                static_cast<std::uint32_t>(seed) ^
                                static_cast<std::uint32_t>(target * 131) ^
                                static_cast<std::uint32_t>(existing));
                            if (existing_hash > worst_hash ||
                                (existing_hash == worst_hash && existing >
                                 frontier_vertices[frontier_base + worst])) {
                                worst = index;
                                worst_hash = existing_hash;
                            }
                        }
                        if (hash < worst_hash ||
                            (hash == worst_hash && neighbor <
                             frontier_vertices[frontier_base + worst])) {
                            frontier_vertices[frontier_base + worst] = neighbor;
                        }
                    }
                }
                atomicExch(&frontier_lock, 0);
            }
        }
        atomicAdd(reinterpret_cast<unsigned long long*>(&edge_visits[candidate]),
                  static_cast<unsigned long long>(local_visits));
        local_visits = 0;
        __syncthreads();
        if (threadIdx.x == 0 && frontier_overflow) ++overflow_rounds;
        __syncthreads();
        if (frontier_count == 0) break;

        for (int index = threadIdx.x; index < frontier_count;
             index += blockDim.x) {
            const auto vertex = frontier_vertices[frontier_base + index];
            std::int64_t target_count = 0;
            std::int64_t source_count = 0;
            std::int64_t internal_count = 0;
            for (std::int64_t edge = offsets[vertex];
                 edge < offsets[vertex + 1]; ++edge) {
                const auto neighbor = neighbors[edge];
                if (neighbor == vertex) continue;
                const int label = labels[neighbor];
                target_count += label == target;
                source_count += label == source;
                internal_count += label == source &&
                    block_contains_vertex(&block_vertices[vertex_base],
                                          current_size, neighbor);
            }
            frontier_scores[frontier_base + index] =
                target_count - source_count + 2 * internal_count;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int index = 0; index < frontier_count; ++index) {
                const auto vertex = frontier_vertices[frontier_base + index];
                const auto score = frontier_scores[frontier_base + index];
                if (next_vertex < 0 || score > next_delta ||
                    (score == next_delta && vertex < next_vertex)) {
                    next_vertex = vertex;
                    next_delta = score;
                }
            }
            if (next_vertex >= 0) {
                block_vertices[vertex_base + current_size] = next_vertex;
                current_gain += next_delta;
                ++current_size;
                if (current_gain > 0 && current_gain > best_gain) {
                    best_gain = current_gain;
                    best_size = current_size;
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        candidate_gains[candidate] = best_gain;
        candidate_sizes[candidate] = best_size;
        candidate_valid[candidate] = best_size > 0 ? 1 : 0;
        frontier_overflows[candidate] = overflow_rounds;
        edge_visits[candidate] += local_visits;
    }
}

__global__ void apply_block_trial_kernel(
    int candidate_count, int max_size, const std::uint8_t* selected_flags,
    const std::int32_t* candidate_sizes, const std::int32_t* candidate_sources,
    const std::int32_t* candidate_targets, const std::int32_t* vertices,
    const std::int32_t* labels, std::int32_t* trial_labels, int* error) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    if (candidate >= candidate_count || !selected_flags[candidate]) return;
    const int size = candidate_sizes[candidate];
    const auto base = static_cast<std::size_t>(candidate) * max_size;
    for (int index = 0; index < size; ++index) {
        const auto vertex = vertices[base + index];
        if (labels[vertex] != candidate_sources[candidate]) {
            atomicExch(error, 1);
        }
        trial_labels[vertex] = candidate_targets[candidate];
    }
}

__global__ void inverse_sqrt_degree_kernel(
    std::int64_t, const std::int64_t*, float*);
__global__ void mark_degree_local_maxima_kernel(
    std::int64_t, int, const std::int64_t*, const std::int32_t*, const float*,
    std::uint8_t*);
__global__ void mark_degree_local_maxima_warp_kernel(
    int, const std::int64_t*, const std::int64_t*, const std::int32_t*,
    const float*, std::uint8_t*);
__global__ void bfs_frontier_expand_kernel(
    int, const std::int64_t*, const std::int64_t*, const std::int32_t*,
    std::int32_t*, std::int64_t*, int*);
__global__ void update_min_distance_kernel(
    std::int64_t, const std::int32_t*, std::int32_t*);
__global__ void distance_seed_key_kernel(
    std::int64_t, bool, const std::int64_t*, const std::uint8_t*,
    const std::int32_t*, unsigned long long*, int, int);
__global__ void distance_seed_vertex_kernel(
    std::int64_t, bool, const std::int64_t*, const std::uint8_t*,
    const std::int32_t*, unsigned long long,
    unsigned long long*, int, int);

}

SingleGPUPartitioner::SingleGPUPartitioner(
    const CSRGraph& graph, const SingleGPUConfig& config,
    const std::vector<std::int32_t>* initial_labels)
    : graph_(graph), config_(config), initial_labels_(initial_labels),
      n_(graph.vertices()), m_(graph.edges()) {
    if (config_.parts <= 1 || config_.parts > 32) {
        throw std::invalid_argument("parts must be 2..32");
    }
    if (config_.seeds_per_part <= 0) {
        throw std::invalid_argument("seeds_per_part must be positive");
    }
    if (config_.grow_rounds <= 0 || config_.refine_rounds < 0 ||
        config_.global_cycles < 0 || config_.field_rounds < 0 ||
        config_.balance_rounds < 0 || config_.polish_rounds < 0) {
        throw std::invalid_argument("round counts must be nonnegative and grow_rounds positive");
    }
    if (config_.vertex_ratio < 1.0f) {
        throw std::invalid_argument("max_vertex_ratio must be at least 1.0");
    }
    if (config_.time_budget_seconds < 0.0) {
        throw std::invalid_argument("time_budget_seconds must be nonnegative");
    }
    if (config_.descent_micro_batch < 0 || config_.descent_micro_rounds < 1) {
        throw std::invalid_argument(
            "descent micro-batch settings must be nonnegative and rounds positive");
    }
    if (config_.incremental_cut_max_moved_ratio < 0.0f ||
        config_.incremental_cut_max_moved_ratio > 1.0f) {
        throw std::invalid_argument(
            "incremental_cut_max_moved_ratio must be in [0, 1]");
    }
    if (config_.neighbor_count_delta_max_moved_ratio < 0.0f ||
        config_.neighbor_count_delta_max_moved_ratio > 1.0f) {
        throw std::invalid_argument(
            "neighbor_count_delta_max_moved_ratio must be in [0, 1]");
    }
    if (config_.structural_warp_degree < 0) {
        throw std::invalid_argument("structural_warp_degree must be nonnegative");
    }
    if (config_.pair_exchange_rounds < 0 || config_.pair_exchange_rounds > 2) {
        throw std::invalid_argument("pair_exchange_rounds must be in 0..2");
    }
    if (config_.pair_top_targets < 1 || config_.pair_top_targets > 2) {
        throw std::invalid_argument("pair_top_targets must be in 1..2");
    }
    if (config_.pair_bucket_limit < 1 || config_.pair_bucket_limit > 32) {
        throw std::invalid_argument("pair_bucket_limit must be in 1..32");
    }
    if (config_.block_max_size < 2 || config_.block_max_size > 16) {
        throw std::invalid_argument("block_max_size must be in 2..16");
    }
    if (config_.block_seeds_per_pair < 1 || config_.block_seeds_per_pair > 8) {
        throw std::invalid_argument("block_seeds_per_pair must be in 1..8");
    }
    if (config_.block_frontier_limit < 1 || config_.block_frontier_limit > 256) {
        throw std::invalid_argument("block_frontier_limit must be in 1..256");
    }
    if (config_.block_rounds < 0 || config_.block_rounds > 2) {
        throw std::invalid_argument("block_rounds must be in 0..2");
    }
    if (static_cast<std::int64_t>(config_.parts) * config_.seeds_per_part > n_) {
        throw std::invalid_argument("parts * seeds_per_part exceeds vertex count");
    }
    const long double average = static_cast<long double>(n_) / config_.parts;
    capacity_ = static_cast<std::int64_t>(std::floor(
        average * config_.vertex_ratio));
    active_capacity_ = capacity_;
    exploration_capacity_ = static_cast<std::int64_t>(std::floor(
        average * (config_.vertex_ratio + 0.10L)));
    std::uint64_t max_degree = 0;
    for (std::int64_t vertex = 0; vertex < n_; ++vertex) {
        max_degree = std::max<std::uint64_t>(
            max_degree,
            static_cast<std::uint64_t>(graph_.offsets()[vertex + 1] -
                                       graph_.offsets()[vertex]));
    }
    gain_abs_bound_ = std::max<std::int64_t>(
        1, static_cast<std::int64_t>(max_degree * 2));
    pair_raw_capacity_ = n_ * static_cast<std::int64_t>(config_.pair_top_targets);
    pair_bucket_count_ = config_.parts * config_.parts;
    pair_slot_count_ = config_.parts * (config_.parts - 1) / 2;
    block_pair_count_ = config_.parts * (config_.parts - 1);
    const auto block_capacity = static_cast<std::int64_t>(block_pair_count_) *
                                config_.block_seeds_per_pair;
    if (block_capacity > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("block candidate capacity exceeds CUDA count range");
    }
    block_candidate_capacity_ = static_cast<int>(block_capacity);
    CUDA_CHECK(cudaSetDevice(0));
    labels_.assign(static_cast<std::size_t>(n_), -1);
    allocate();
    std::cout << "execution=single_gpu full_csr=1 distributed_mechanisms=0"
              << " gain_bound=" << gain_abs_bound_
              << " maximum_load=" << capacity_
              << " exploration_maximum_load=" << exploration_capacity_
              << " high_degree_vertices=" << high_degree_count_ << '\n';
}

SingleGPUPartitioner::~SingleGPUPartitioner() { release(); }

void SingleGPUPartitioner::allocate() {
    const auto vertices = static_cast<std::size_t>(n_);
    const auto edges = static_cast<std::size_t>(m_);
    const auto parts = static_cast<std::size_t>(config_.parts);
    CUDA_CHECK(cudaMalloc(&d_offsets_, (vertices + 1) * sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_neighbors_, edges * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_labels_, vertices * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_labels_, vertices * sizeof(std::int32_t)));
    if (config_.time_budget_seconds > 0.0) {
        CUDA_CHECK(cudaMalloc(&d_time_checkpoint_labels_,
                              vertices * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_time_checkpoint_best_labels_,
                              vertices * sizeof(std::int32_t)));
    }
    CUDA_CHECK(cudaMalloc(&d_targets_, vertices * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_gains_, vertices * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_best_labels_, vertices * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_old_labels_, vertices * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_neighbor_label_counts_,
                          vertices * parts * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_core_flags_, vertices * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_candidates_, vertices * sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted_candidates_, vertices * sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_high_degree_vertices_,
                          vertices * sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_keys_, vertices * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sorted_keys_, vertices * sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_loads_, parts * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_load_deltas_, parts * parts * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_gain_hist_, parts * kGainBins * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_target_quota_, parts * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_source_quota_, parts * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_target_begin_, parts * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_inv_sqrt_degree_, vertices * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_signal_, vertices * parts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_next_signal_, vertices * parts * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cut_, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_candidate_count_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_changed_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_moved_flags_, vertices * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_cut_delta_, sizeof(std::int64_t)));
    const auto pair_elements = static_cast<std::size_t>(pair_raw_capacity_);
    CUDA_CHECK(cudaMalloc(
        &d_pair_targets_, pair_elements * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(
        &d_pair_gains_, pair_elements * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_pair_keys_,
                          static_cast<std::size_t>(pair_raw_capacity_) *
                              sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_pair_sorted_keys_,
                          static_cast<std::size_t>(pair_raw_capacity_) *
                              sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_pair_values_,
                          static_cast<std::size_t>(pair_raw_capacity_) *
                              sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_pair_sorted_values_,
                          static_cast<std::size_t>(pair_raw_capacity_) *
                              sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(
        &d_pair_bucket_counts_,
        static_cast<std::size_t>(pair_bucket_count_) * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(
        &d_pair_bucket_begin_,
        static_cast<std::size_t>(pair_bucket_count_ + 1) * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMalloc(&d_pair_u_,
                          static_cast<std::size_t>(pair_slot_count_) *
                              sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_pair_v_,
                          static_cast<std::size_t>(pair_slot_count_) *
                              sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_pair_result_gains_,
                          static_cast<std::size_t>(pair_slot_count_) *
                              sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_pair_proposed_flags_,
                          static_cast<std::size_t>(pair_slot_count_) *
                              sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_pair_accepted_flags_,
                          static_cast<std::size_t>(pair_slot_count_) *
                              sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMalloc(
        &d_pair_moved_vertices_,
        static_cast<std::size_t>(pair_slot_count_) * 2 * sizeof(std::int64_t)));
    CUDA_CHECK(cudaMalloc(&d_pair_moved_count_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_boundary_count_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_accepted_count_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pair_accepted_gain_, sizeof(std::int64_t)));

    if (config_.block_lp) {
        const auto block_candidates =
            static_cast<std::size_t>(block_candidate_capacity_);
        const auto block_vertices = block_candidates *
                                    static_cast<std::size_t>(config_.block_max_size);
        const auto block_frontier = block_candidates *
                                    static_cast<std::size_t>(config_.block_frontier_limit);
        CUDA_CHECK(cudaMalloc(&d_block_seed_vertices_,
                              block_candidates * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_block_seed_gains_,
                              block_candidates * sizeof(std::int64_t)));
        CUDA_CHECK(cudaMalloc(&d_block_seed_sources_,
                              block_candidates * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_block_seed_targets_,
                              block_candidates * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_block_vertices_,
                              block_vertices * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_block_candidate_gains_,
                              block_candidates * sizeof(std::int64_t)));
        CUDA_CHECK(cudaMalloc(&d_block_candidate_sizes_,
                              block_candidates * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_block_candidate_valid_,
                              block_candidates * sizeof(std::uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_block_frontier_vertices_,
                              block_frontier * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_block_frontier_scores_,
                              block_frontier * sizeof(std::int64_t)));
        CUDA_CHECK(cudaMalloc(&d_block_selected_flags_,
                              block_candidates * sizeof(std::uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_block_trial_labels_,
                              vertices * sizeof(std::int32_t)));
        CUDA_CHECK(cudaMalloc(&d_block_frontier_overflows_,
                              block_candidates * sizeof(std::uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_block_edge_visits_,
                              block_candidates * sizeof(std::uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_block_error_, sizeof(int)));
    }

    const cub::CountingInputIterator<std::int64_t> counting(0);
    const MovePredicate predicate{d_targets_, d_gains_, d_labels_, d_loads_,
                                  capacity_, config_.min_gain, config_.repair};
    CUDA_CHECK(cub::DeviceSelect::If(
        nullptr, select_temp_bytes_, counting, d_candidates_, d_candidate_count_,
        static_cast<int>(n_), predicate));
    if (pair_raw_capacity_ > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("pair candidate capacity exceeds CUDA count range");
    }
    const PairCandidatePredicate pair_predicate{d_pair_targets_,
                                                config_.pair_top_targets};
    std::size_t pair_select_temp_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::If(
        nullptr, pair_select_temp_bytes, counting, d_pair_values_,
        d_candidate_count_, static_cast<int>(pair_raw_capacity_),
        pair_predicate));
    select_temp_bytes_ = std::max(select_temp_bytes_, pair_select_temp_bytes);
    if (config_.structural_warp_degree > 0) {
        std::size_t degree_select_bytes = 0;
        const DegreePredicate degree_predicate{
            d_offsets_, config_.structural_warp_degree};
        CUDA_CHECK(cub::DeviceSelect::If(
            nullptr, degree_select_bytes, counting, d_high_degree_vertices_,
            d_candidate_count_, static_cast<int>(n_), degree_predicate));
        select_temp_bytes_ = std::max(select_temp_bytes_, degree_select_bytes);
    }
    CUDA_CHECK(cudaMalloc(&d_select_temp_, select_temp_bytes_));
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_temp_bytes_, d_keys_, d_sorted_keys_, d_candidates_,
        d_sorted_candidates_, static_cast<int>(n_), 0, 64));
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, pair_sort_temp_bytes_, d_pair_keys_, d_pair_sorted_keys_,
        d_pair_values_, d_pair_sorted_values_, static_cast<int>(pair_raw_capacity_),
        0, 32));
    CUDA_CHECK(cudaMalloc(&d_sort_temp_, sort_temp_bytes_));
    CUDA_CHECK(cudaMalloc(&d_pair_sort_temp_, pair_sort_temp_bytes_));

    CUDA_CHECK(cudaMemcpy(d_offsets_, graph_.offsets().data(),
                          (vertices + 1) * sizeof(std::int64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors_, graph_.neighbors().data(),
                          edges * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));
    if (config_.structural_warp_degree > 0) {
        const DegreePredicate degree_predicate{
            d_offsets_, config_.structural_warp_degree};
        CUDA_CHECK(cub::DeviceSelect::If(
            d_select_temp_, select_temp_bytes_, counting,
            d_high_degree_vertices_, d_candidate_count_,
            static_cast<int>(n_), degree_predicate));
        CUDA_CHECK(cudaMemcpy(&high_degree_count_, d_candidate_count_,
                              sizeof(int), cudaMemcpyDeviceToHost));
    }
}

void SingleGPUPartitioner::release() noexcept {
    cudaFree(d_offsets_);
    cudaFree(d_neighbors_);
    cudaFree(d_labels_);
    cudaFree(d_next_labels_);
    cudaFree(d_time_checkpoint_labels_);
    cudaFree(d_time_checkpoint_best_labels_);
    cudaFree(d_targets_);
    cudaFree(d_gains_);
    cudaFree(d_best_labels_);
    cudaFree(d_old_labels_);
    cudaFree(d_neighbor_label_counts_);
    cudaFree(d_core_flags_);
    cudaFree(d_candidates_);
    cudaFree(d_sorted_candidates_);
    cudaFree(d_high_degree_vertices_);
    cudaFree(d_keys_);
    cudaFree(d_sorted_keys_);
    cudaFree(d_loads_);
    cudaFree(d_load_deltas_);
    cudaFree(d_gain_hist_);
    cudaFree(d_target_quota_);
    cudaFree(d_source_quota_);
    cudaFree(d_target_begin_);
    cudaFree(d_inv_sqrt_degree_);
    cudaFree(d_signal_);
    cudaFree(d_next_signal_);
    cudaFree(d_cut_);
    cudaFree(d_candidate_count_);
    cudaFree(d_changed_);
    cudaFree(d_moved_flags_);
    cudaFree(d_cut_delta_);
    cudaFree(d_pair_targets_);
    cudaFree(d_pair_gains_);
    cudaFree(d_pair_keys_);
    cudaFree(d_pair_sorted_keys_);
    cudaFree(d_pair_values_);
    cudaFree(d_pair_sorted_values_);
    cudaFree(d_pair_bucket_counts_);
    cudaFree(d_pair_bucket_begin_);
    cudaFree(d_pair_u_);
    cudaFree(d_pair_v_);
    cudaFree(d_pair_result_gains_);
    cudaFree(d_pair_proposed_flags_);
    cudaFree(d_pair_accepted_flags_);
    cudaFree(d_pair_moved_vertices_);
    cudaFree(d_pair_moved_count_);
    cudaFree(d_pair_boundary_count_);
    cudaFree(d_pair_accepted_count_);
    cudaFree(d_pair_accepted_gain_);
    cudaFree(d_block_seed_vertices_);
    cudaFree(d_block_seed_gains_);
    cudaFree(d_block_seed_sources_);
    cudaFree(d_block_seed_targets_);
    cudaFree(d_block_vertices_);
    cudaFree(d_block_candidate_gains_);
    cudaFree(d_block_candidate_sizes_);
    cudaFree(d_block_candidate_valid_);
    cudaFree(d_block_frontier_vertices_);
    cudaFree(d_block_frontier_scores_);
    cudaFree(d_block_selected_flags_);
    cudaFree(d_block_trial_labels_);
    cudaFree(d_block_frontier_overflows_);
    cudaFree(d_block_edge_visits_);
    cudaFree(d_block_error_);
    cudaFree(d_select_temp_);
    cudaFree(d_sort_temp_);
    cudaFree(d_pair_sort_temp_);
}

void SingleGPUPartitioner::build_degree_normalization() {
    const auto start = std::chrono::steady_clock::now();
    const int blocks = static_cast<int>((n_ + 255) / 256);
    inverse_sqrt_degree_kernel<<<blocks, 256>>>(
        n_, d_offsets_, d_inv_sqrt_degree_);
    CUDA_CHECK(cudaGetLastError());
    mark_degree_local_maxima_kernel<<<blocks, 256>>>(
        n_, high_degree_count_ > 0 ? config_.structural_warp_degree : 0,
        d_offsets_, d_neighbors_, d_inv_sqrt_degree_, d_core_flags_);
    if (high_degree_count_ > 0) {
        mark_degree_local_maxima_warp_kernel<<<
            (high_degree_count_ * 32 + 255) / 256, 256>>>(
            high_degree_count_, d_high_degree_vertices_, d_offsets_,
            d_neighbors_, d_inv_sqrt_degree_, d_core_flags_);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "structural_init=degree_normalization_and_core_flags seconds="
              << seconds << '\n';
}

void SingleGPUPartitioner::distance_bfs(std::int64_t source) {
    CUDA_CHECK(cudaMemset(d_labels_, 0x3f,
                          static_cast<std::size_t>(n_) * sizeof(std::int32_t)));
    const std::int32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_labels_ + source, &zero, sizeof(zero),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_candidates_, &source, sizeof(source),
                          cudaMemcpyHostToDevice));
    std::int64_t* current_frontier = d_candidates_;
    std::int64_t* next_frontier = d_sorted_candidates_;
    int frontier_size = 1;
    while (frontier_size > 0) {
        CUDA_CHECK(cudaMemset(d_candidate_count_, 0, sizeof(int)));
        bfs_frontier_expand_kernel<<<(frontier_size * 32LL + 255) / 256, 256>>>(
            frontier_size, current_frontier, d_offsets_, d_neighbors_,
            d_labels_, next_frontier, d_candidate_count_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&frontier_size, d_candidate_count_, sizeof(int),
                              cudaMemcpyDeviceToHost));
        std::swap(current_frontier, next_frontier);
    }
}

void SingleGPUPartitioner::choose_distance_separated_seeds() {
    seeds_.clear();
    const int seed_count = config_.parts * config_.seeds_per_part;
    seeds_.reserve(static_cast<std::size_t>(seed_count));
    CUDA_CHECK(cudaMemset(d_gains_, 0x3f,
                          static_cast<std::size_t>(n_) * sizeof(std::int32_t)));
    const int blocks = static_cast<int>((n_ + 255) / 256);
    for (int selected = 0; selected < seed_count; ++selected) {
        if (selected > 0) {
            distance_bfs(seeds_.back());
            update_min_distance_kernel<<<blocks, 256>>>(
                n_, d_labels_, d_gains_);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemset(d_cut_, 0, sizeof(unsigned long long)));
        distance_seed_key_kernel<<<blocks, 256>>>(
            n_, selected == 0, d_offsets_, d_core_flags_, d_gains_, d_cut_, config_.search_seed, config_.seed_distance_power);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long best_key = 0;
        CUDA_CHECK(cudaMemcpy(&best_key, d_cut_, sizeof(best_key),
                              cudaMemcpyDeviceToHost));
        const unsigned long long no_vertex = ~0ULL;
        CUDA_CHECK(cudaMemcpy(d_candidates_, &no_vertex, sizeof(no_vertex),
                              cudaMemcpyHostToDevice));
        distance_seed_vertex_kernel<<<blocks, 256>>>(
            n_, selected == 0, d_offsets_, d_core_flags_, d_gains_, best_key,
            reinterpret_cast<unsigned long long*>(d_candidates_), config_.search_seed, config_.seed_distance_power);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long best_vertex = no_vertex;
        CUDA_CHECK(cudaMemcpy(&best_vertex, d_candidates_, sizeof(best_vertex),
                              cudaMemcpyDeviceToHost));
        if (best_vertex == no_vertex) {
            throw std::runtime_error("distance seed selection found no core vertex");
        }
        seeds_.push_back(static_cast<std::int64_t>(best_vertex));
        std::cout << "distance_seed index=" << selected
                  << " part=" << (selected % config_.parts)
                  << " vertex=" << best_vertex
                  << " separation_score=" << (best_key >> 32)
                  << " degree=" << static_cast<std::uint32_t>(best_key) << '\n';
    }
}

namespace {

struct CandidatePredicate {
    const std::int32_t* targets;

    __device__ bool operator()(const std::int64_t& vertex) const {
        return targets[vertex] >= 0;
    }
};

__global__ void inverse_sqrt_degree_kernel(
    std::int64_t n, const std::int64_t* offsets, float* inverse_sqrt_degree) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex >= n) return;
    const float degree = static_cast<float>(offsets[vertex + 1] - offsets[vertex]);
    inverse_sqrt_degree[vertex] = degree > 0.0f ? rsqrtf(degree) : 0.0f;
}

__global__ void bfs_frontier_expand_kernel(
    int frontier_size, const std::int64_t* frontier,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    std::int32_t* distance, std::int64_t* next_frontier,
    int* next_frontier_size) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= frontier_size) return;
    const auto vertex = frontier[warp];
    const std::int32_t next_distance = distance[vertex] + 1;
    for (std::int64_t edge = offsets[vertex] + lane;
         edge < offsets[vertex + 1]; edge += 32) {
        const auto neighbor = neighbors[edge];
        if (atomicCAS(&distance[neighbor], kBfsInf, next_distance) == kBfsInf) {
            const int output = atomicAdd(next_frontier_size, 1);
            next_frontier[output] = neighbor;
        }
    }
}

__global__ void update_min_distance_kernel(
    std::int64_t n, const std::int32_t* distance,
    std::int32_t* min_distance) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex < n) min_distance[vertex] = min(min_distance[vertex], distance[vertex]);
}

__device__ __forceinline__ bool degree_local_maximum(
    std::int64_t vertex, const std::int64_t* offsets,
    const std::int32_t* neighbors, const float* inverse_sqrt_degree) {
    const auto degree = offsets[vertex + 1] - offsets[vertex];
    if (degree == 0) return false;
    const float own = inverse_sqrt_degree[vertex];
    for (std::int64_t edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
        const float neighbor = inverse_sqrt_degree[neighbors[edge]];
        if (neighbor > 0.0f && neighbor < own) return false;
    }
    return true;
}

__global__ void mark_degree_local_maxima_kernel(
    std::int64_t n, int warp_degree, const std::int64_t* offsets,
    const std::int32_t* neighbors, const float* inverse_sqrt_degree,
    std::uint8_t* core_flags) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (vertex >= n) return;
    if (warp_degree > 0 &&
        offsets[vertex + 1] - offsets[vertex] >= warp_degree) return;
    core_flags[vertex] = degree_local_maximum(
        vertex, offsets, neighbors, inverse_sqrt_degree);
}

__global__ void mark_degree_local_maxima_warp_kernel(
    int vertex_count, const std::int64_t* vertices,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const float* inverse_sqrt_degree, std::uint8_t* core_flags) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= vertex_count) return;
    const auto vertex = vertices[warp];
    const float own = inverse_sqrt_degree[vertex];
    bool local_is_maximum = true;
    for (std::int64_t edge = offsets[vertex] + lane;
         edge < offsets[vertex + 1]; edge += 32) {
        const float neighbor = inverse_sqrt_degree[neighbors[edge]];
        if (neighbor > 0.0f && neighbor < own) local_is_maximum = false;
    }
    const bool is_maximum = __all_sync(0xffffffffu, local_is_maximum);
    if (lane == 0) core_flags[vertex] = is_maximum;
}

__global__ void distance_seed_key_kernel(
    std::int64_t n, bool first, const std::int64_t* offsets,
    const std::uint8_t* core_flags,
    const std::int32_t* min_distance, unsigned long long* best_key, int search_seed, int distance_power) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex >= n || !core_flags[vertex]) return;
    if (!first && min_distance[vertex] == kBfsInf) return;
    const auto degree = static_cast<std::uint32_t>(
        offsets[vertex + 1] - offsets[vertex]);
    const auto distance = first ? 0U : static_cast<std::uint32_t>(min_distance[vertex]);
    unsigned long long distance_score=distance;
    for(int p=1;p<distance_power;++p)distance_score=min(0xffffffffULL,distance_score*distance);
    auto primary = first
        ? static_cast<unsigned long long>(degree)
        : min(0xffffffffULL, distance_score * degree);
    if(search_seed) primary = primary * (1ULL + (hash32(static_cast<unsigned>(vertex) ^ static_cast<unsigned>(search_seed)) & 1023U)) / 1024ULL;
    atomicMax(best_key, (primary << 32) | degree);
}

__global__ void distance_seed_vertex_kernel(
    std::int64_t n, bool first, const std::int64_t* offsets,
    const std::uint8_t* core_flags,
    const std::int32_t* min_distance, unsigned long long best_key,
    unsigned long long* best_vertex, int search_seed, int distance_power) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex >= n || !core_flags[vertex]) return;
    if (!first && min_distance[vertex] == kBfsInf) return;
    const auto degree = static_cast<std::uint32_t>(
        offsets[vertex + 1] - offsets[vertex]);
    const auto distance = first ? 0U : static_cast<std::uint32_t>(min_distance[vertex]);
    unsigned long long distance_score=distance;
    for(int p=1;p<distance_power;++p)distance_score=min(0xffffffffULL,distance_score*distance);
    auto primary = first
        ? static_cast<unsigned long long>(degree)
        : min(0xffffffffULL, distance_score * degree);
    if(search_seed) primary = primary * (1ULL + (hash32(static_cast<unsigned>(vertex) ^ static_cast<unsigned>(search_seed)) & 1023U)) / 1024ULL;
    if (((primary << 32) | degree) == best_key) {
        atomicMin(best_vertex, static_cast<unsigned long long>(vertex));
    }
}

__global__ void store_distance_channel_kernel(
    std::int64_t n, int parts, int channel,
    bool merge_minimum, const std::int32_t* distance, float* matrix) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex < n) {
        auto& value = matrix[
            vertex * static_cast<std::int64_t>(parts) + channel];
        const float candidate = static_cast<float>(distance[vertex]);
        value = merge_minimum ? fminf(value, candidate) : candidate;
    }
}

__global__ void distance_capacity_propose_kernel(
    std::int64_t n, int parts, const float* distance_matrix,
    const std::int32_t* assignment, const unsigned long long* loads,
    std::int64_t capacity, std::int32_t* targets, std::int32_t* gains) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex >= n) return;
    if (assignment[vertex] >= 0) {
        targets[vertex] = -1;
        gains[vertex] = 0;
        return;
    }
    const auto base = static_cast<std::size_t>(vertex) * parts;
    int best = -1;
    float best_distance = 1.0e30f;
    std::uint32_t best_tie = 0;
    for (int part = 0; part < parts; ++part) {
        if (loads[part] >= static_cast<unsigned long long>(capacity)) continue;
        const float distance = distance_matrix[base + part];
        const auto tie = hash32(static_cast<std::uint32_t>(vertex) ^
                                static_cast<std::uint32_t>(part * 131));
        if (best < 0 || distance < best_distance ||
            (distance == best_distance && tie > best_tie)) {
            best = part;
            best_distance = distance;
            best_tie = tie;
        }
    }
    targets[vertex] = best;
    const float score = best_distance >= 1.0e20f ? -1000000.0f : -best_distance;
    gains[vertex] = best < 0 ? 0 : max(-1000000, min(1000000,
        static_cast<int>(score)));
}

__global__ void count_distance_hist_kernel(
    int count, const std::int64_t* candidates, const std::int32_t* targets,
    const std::int32_t* gains, unsigned long long* histogram) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto vertex = candidates[index];
    const int target = targets[vertex];
    const int distance = min(kGainBins - 1, max(0, -gains[vertex]));
    atomicAdd(&histogram[static_cast<std::size_t>(target) * kGainBins + distance], 1ULL);
}

__global__ void make_distance_keys_kernel(
    int count, const std::int64_t* candidates, const std::int32_t* targets,
    const std::int32_t* gains, std::uint64_t* keys) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto vertex = candidates[index];
    const auto target = static_cast<std::uint64_t>(targets[vertex]);
    const auto distance = static_cast<std::uint64_t>(
        min(kGainBins - 1, max(0, -gains[vertex])));
    keys[index] = (target << 59) | (distance << 32) |
                  (static_cast<std::uint64_t>(vertex) & 0xffffffffULL);
}

__global__ void apply_initial_quota_kernel(
    int count, const std::int64_t* candidates, const std::int32_t* targets,
    const std::uint32_t* quota, const std::uint32_t* target_begin,
    std::int32_t* assignment, int* changed) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto vertex = candidates[index];
    const int target = targets[vertex];
    const auto rank = static_cast<std::uint32_t>(index) - target_begin[target];
    if (rank < quota[target] && assignment[vertex] < 0) {
        assignment[vertex] = target;
        atomicAdd(changed, 1);
    }
}

__global__ void label_probe_kernel(
    std::int64_t n, int parts, const std::int32_t* labels, float* signal) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex >= n) return;
    const int label = labels[vertex];
    const auto base = static_cast<std::size_t>(vertex) * parts;
    const float baseline = 1.0f / static_cast<float>(parts);
    for (int part = 0; part < parts; ++part) {
        signal[base + part] = label == part ? 1.0f - baseline : -baseline;
    }
}

template <int FixedParts>
__global__ void structural_diffusion_kernel(
    std::int64_t n, int runtime_parts, int thread_degree_limit,
    const std::int64_t* offsets,
    const std::int32_t* neighbors, const float* inverse_sqrt_degree,
    const float* signal, float* next_signal) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex >= n) return;
    constexpr int storage_parts = FixedParts == 0 ? 32 : FixedParts;
    const int parts = FixedParts == 0 ? runtime_parts : FixedParts;
    const auto degree = offsets[vertex + 1] - offsets[vertex];
    if (thread_degree_limit > 0 && degree >= thread_degree_limit) return;
    float sums[storage_parts] = {};
    const float source_inv = inverse_sqrt_degree[vertex];
    if (source_inv > 0.0f) {
        for (std::int64_t edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
            const auto neighbor = neighbors[edge];
            const float neighbor_inv = inverse_sqrt_degree[neighbor];
            if (neighbor_inv <= 0.0f) continue;
            const float scale = source_inv * neighbor_inv;
            if constexpr (FixedParts == 4) {
                const float4 neighbor_signal =
                    reinterpret_cast<const float4*>(signal)[neighbor];
                sums[0] += neighbor_signal.x * scale;
                sums[1] += neighbor_signal.y * scale;
                sums[2] += neighbor_signal.z * scale;
                sums[3] += neighbor_signal.w * scale;
            } else {
                for (int part = 0; part < parts; ++part) {
                    sums[part] += signal[
                        neighbor * static_cast<std::int64_t>(parts) + part] *
                        scale;
                }
            }
        }
    }
    const auto base = static_cast<std::size_t>(vertex) * parts;
    if constexpr (FixedParts == 4) {
        const float4 current = reinterpret_cast<const float4*>(signal)[vertex];
        reinterpret_cast<float4*>(next_signal)[vertex] = make_float4(
            0.5f * current.x + 0.5f * sums[0],
            0.5f * current.y + 0.5f * sums[1],
            0.5f * current.z + 0.5f * sums[2],
            0.5f * current.w + 0.5f * sums[3]);
    } else {
        for (int part = 0; part < parts; ++part) {
            next_signal[base + part] =
                0.5f * signal[base + part] + 0.5f * sums[part];
        }
    }
}

__global__ void structural_diffusion_warp4_kernel(
    int vertex_count, const std::int64_t* vertices,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const float* inverse_sqrt_degree, const float* signal,
    float* next_signal) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= vertex_count) return;
    const auto vertex = vertices[warp];
    float sums[4] = {};
    const float source_inv = inverse_sqrt_degree[vertex];
    for (std::int64_t edge = offsets[vertex] + lane;
         edge < offsets[vertex + 1]; edge += 32) {
        const auto neighbor = neighbors[edge];
        const float neighbor_inv = inverse_sqrt_degree[neighbor];
        if (neighbor_inv <= 0.0f) continue;
        const float scale = source_inv * neighbor_inv;
        const float4 neighbor_signal =
            reinterpret_cast<const float4*>(signal)[neighbor];
        sums[0] += neighbor_signal.x * scale;
        sums[1] += neighbor_signal.y * scale;
        sums[2] += neighbor_signal.z * scale;
        sums[3] += neighbor_signal.w * scale;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            sums[part] += __shfl_down_sync(0xffffffffU, sums[part], offset);
        }
    }
    if (lane == 0) {
        const float4 current = reinterpret_cast<const float4*>(signal)[vertex];
        reinterpret_cast<float4*>(next_signal)[vertex] = make_float4(
            0.5f * current.x + 0.5f * sums[0],
            0.5f * current.y + 0.5f * sums[1],
            0.5f * current.z + 0.5f * sums[2],
            0.5f * current.w + 0.5f * sums[3]);
    }
}

__global__ void field_target_kernel(
    std::int64_t n, int parts, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::int32_t* labels,
    const float* signal, std::int32_t* targets, std::int32_t* gains) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex >= n) return;
    const auto base = static_cast<std::size_t>(vertex) * parts;
    float best_score = -1.0e30f;
    float second_score = -1.0e30f;
    int best_part = 0;
    std::uint32_t best_tie = 0;
    for (int part = 0; part < parts; ++part) {
        const float score = signal[base + part];
        const auto tie = hash32(static_cast<std::uint32_t>(vertex) ^
                                static_cast<std::uint32_t>(part * 131));
        if (score > best_score || (score == best_score && tie > best_tie)) {
            second_score = best_score;
            best_score = score;
            best_part = part;
            best_tie = tie;
        } else if (score > second_score) {
            second_score = score;
        }
    }
    const int current = labels[vertex];
    bool touches_target = best_part == current;
    for (std::int64_t edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
        if (labels[neighbors[edge]] == best_part) touches_target = true;
    }
    targets[vertex] = touches_target ? best_part : current;
    const float margin = fmaxf(0.0f, best_score - second_score);
    gains[vertex] = touches_target
        ? min(static_cast<int>(kFieldGainBound),
              static_cast<int>(margin * static_cast<float>(kFieldGainBound)))
        : 0;
}

__global__ void count_labels_kernel(
    std::int64_t n, int parts, const std::int32_t* labels,
    unsigned long long* counts) {
    extern __shared__ unsigned long long block_counts[];
    for (int part = threadIdx.x; part < parts; part += blockDim.x) {
        block_counts[part] = 0;
    }
    __syncthreads();
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex < n) {
        const int part = labels[vertex];
        if (part >= 0 && part < parts) atomicAdd(&block_counts[part], 1ULL);
    }
    __syncthreads();
    for (int part = threadIdx.x; part < parts; part += blockDim.x) {
        if (block_counts[part]) atomicAdd(&counts[part], block_counts[part]);
    }
}

template <int FixedParts>
__global__ void build_neighbor_label_counts_kernel(
    std::int64_t n, int runtime_parts, int warp_degree,
    const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::int32_t* labels,
    std::int32_t* neighbor_label_counts) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (vertex >= n) return;
    if (warp_degree > 0 && offsets[vertex + 1] - offsets[vertex] >= warp_degree) return;
    constexpr int storage_parts = FixedParts == 0 ? 32 : FixedParts;
    const int parts = FixedParts == 0 ? runtime_parts : FixedParts;
    int counts[storage_parts] = {};
    for (std::int64_t edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
        const int part = labels[neighbors[edge]];
        if (part >= 0 && part < parts) ++counts[part];
    }
    const auto base = static_cast<std::size_t>(vertex) * parts;
    for (int part = 0; part < parts; ++part) {
        neighbor_label_counts[base + part] = counts[part];
    }
}

__global__ void build_neighbor_label_counts_warp4_kernel(
    int vertex_count, const std::int64_t* vertices,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::int32_t* labels, std::int32_t* neighbor_label_counts) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= vertex_count) return;
    const auto vertex = vertices[warp];
    int counts[4] = {};
    for (std::int64_t edge = offsets[vertex] + lane;
         edge < offsets[vertex + 1]; edge += 32) {
        const int part = labels[neighbors[edge]];
        if (part >= 0 && part < 4) ++counts[part];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            counts[part] += __shfl_down_sync(0xffffffffu, counts[part], offset);
        }
    }
    if (lane == 0) {
        const auto base = static_cast<std::size_t>(vertex) * 4;
#pragma unroll
        for (int part = 0; part < 4; ++part) {
            neighbor_label_counts[base + part] = counts[part];
        }
    }
}

__global__ void update_neighbor_label_counts_warp_kernel(
    int moved_count, int parts, const std::int64_t* moved_vertices,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::int32_t* labels, const std::int32_t* old_labels,
    std::int32_t* neighbor_label_counts) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= moved_count) return;
    const auto vertex = moved_vertices[warp];
    const int old_part = old_labels[vertex];
    const int new_part = labels[vertex];
    if (old_part == new_part) return;
    for (std::int64_t edge = offsets[vertex] + lane;
         edge < offsets[vertex + 1]; edge += 32) {
        const auto neighbor = neighbors[edge];
        const auto base = static_cast<std::size_t>(neighbor) * parts;
        atomicSub(&neighbor_label_counts[base + old_part], 1);
        atomicAdd(&neighbor_label_counts[base + new_part], 1);
    }
}

template <int FixedParts>
__global__ void pair_candidate_kernel(
    std::int64_t n, int runtime_parts, int top_targets,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::int32_t* neighbor_label_counts, const std::int32_t* labels,
    std::int32_t* pair_targets, std::int32_t* pair_gains,
    int* boundary_count) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (vertex >= n) return;
    constexpr int storage_parts = FixedParts == 0 ? 32 : FixedParts;
    const int parts = FixedParts == 0 ? runtime_parts : FixedParts;
    int counts[storage_parts] = {};
    if (neighbor_label_counts) {
        const auto base = static_cast<std::size_t>(vertex) * parts;
        for (int part = 0; part < parts; ++part) {
            counts[part] = neighbor_label_counts[base + part];
        }
    } else {
        for (std::int64_t edge = offsets[vertex];
             edge < offsets[vertex + 1]; ++edge) {
            const int part = labels[neighbors[edge]];
            if (part >= 0 && part < parts) ++counts[part];
        }
    }

    const int current = labels[vertex];
    const auto base = static_cast<std::size_t>(vertex) * top_targets;
    for (int slot = 0; slot < top_targets; ++slot) {
        int best_target = -1;
        int best_gain = -0x3f3f3f3f;
        for (int part = 0; part < parts; ++part) {
            if (part == current || counts[part] == 0) continue;
            const int gain = counts[part] - counts[current];
            if (best_target < 0 || gain > best_gain ||
                (gain == best_gain && part < best_target)) {
                best_target = part;
                best_gain = gain;
            }
        }
        pair_targets[base + slot] = best_target;
        pair_gains[base + slot] = best_target < 0 ? 0 : best_gain;
        if (best_target >= 0) counts[best_target] = 0;
    }
    if (pair_targets[base] >= 0) atomicAdd(boundary_count, 1);
}

__global__ void make_pair_sort_keys_kernel(
    int count, int stage, int parts, int top_targets,
    const std::int64_t* values, const std::int32_t* targets,
    const std::int32_t* gains, const std::int32_t* labels,
    std::uint64_t* keys) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto value = values[index];
    const auto vertex = value / top_targets;
    const auto slot = value - vertex * top_targets;
    if (stage == 0) {
        keys[index] = static_cast<std::uint64_t>(vertex);
    } else if (stage == 1) {
        const auto gain = static_cast<std::uint32_t>(
            gains[vertex * top_targets + slot]);
        keys[index] = static_cast<std::uint64_t>(
            ~(gain ^ 0x80000000U));
    } else {
        const int source = labels[vertex];
        const int target = targets[vertex * top_targets + slot];
        keys[index] = static_cast<std::uint64_t>(source * parts + target);
    }
}

__global__ void count_pair_buckets_kernel(
    int count, int parts, int top_targets, const std::int64_t* values,
    const std::int32_t* targets, const std::int32_t* labels,
    std::int32_t* bucket_counts) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto value = values[index];
    const auto vertex = value / top_targets;
    const auto slot = value - vertex * top_targets;
    const int source = labels[vertex];
    const int target = targets[vertex * top_targets + slot];
    atomicAdd(&bucket_counts[source * parts + target], 1);
}

__device__ __forceinline__ std::int64_t pair_edge_multiplicity(
    std::int64_t source, std::int64_t target,
    const std::int64_t* offsets, const std::int32_t* neighbors) {
    std::int64_t multiplicity = 0;
    for (std::int64_t edge = offsets[source]; edge < offsets[source + 1]; ++edge) {
        multiplicity += neighbors[edge] == target;
    }
    return multiplicity;
}

__device__ __forceinline__ bool pair_edge_exists(
    std::int64_t left, std::int64_t right,
    const std::int64_t* offsets, const std::int32_t* neighbors) {
    return pair_edge_multiplicity(left, right, offsets, neighbors) > 0 ||
           pair_edge_multiplicity(right, left, offsets, neighbors) > 0;
}

__device__ __forceinline__ bool pair_priority_better(
    std::int64_t gain, std::int64_t left, std::int64_t right,
    std::int64_t other_gain, std::int64_t other_left,
    std::int64_t other_right) {
    if (gain != other_gain) return gain > other_gain;
    if (left != other_left) return left < other_left;
    return right < other_right;
}

__global__ void pair_exchange_kernel(
    int pair_slots, int parts, int top_targets, int bucket_limit,
    const std::int32_t* bucket_begin, const std::int32_t* bucket_counts,
    const std::int64_t* sorted_values, const std::int32_t* targets,
    const std::int32_t* gains, const std::int32_t* labels,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    std::int64_t* pair_u, std::int64_t* pair_v,
    std::int64_t* pair_result_gains, std::uint8_t* proposed_flags) {
    const int pair_slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_slot >= pair_slots) return;

    int left_part = -1;
    int right_part = -1;
    int seen = 0;
    for (int source = 0; source < parts; ++source) {
        for (int target = source + 1; target < parts; ++target) {
            if (seen == pair_slot) {
                left_part = source;
                right_part = target;
            }
            ++seen;
        }
    }

    const int forward_bucket = left_part * parts + right_part;
    const int reverse_bucket = right_part * parts + left_part;
    const int forward_count = min(bucket_counts[forward_bucket], bucket_limit);
    const int reverse_count = min(bucket_counts[reverse_bucket], bucket_limit);
    const int forward_begin = bucket_begin[forward_bucket];
    const int reverse_begin = bucket_begin[reverse_bucket];

    bool found = false;
    std::int64_t best_gain = 0;
    std::int64_t best_u = -1;
    std::int64_t best_v = -1;
    for (int left_index = 0; left_index < forward_count; ++left_index) {
        const auto left_value = sorted_values[forward_begin + left_index];
        const auto u = left_value / top_targets;
        const auto u_slot = left_value - u * top_targets;
        const auto gain_u = static_cast<std::int64_t>(
            gains[u * top_targets + u_slot]);
        for (int right_index = 0; right_index < reverse_count; ++right_index) {
            const auto right_value = sorted_values[reverse_begin + right_index];
            const auto v = right_value / top_targets;
            const auto v_slot = right_value - v * top_targets;
            const auto gain_v = static_cast<std::int64_t>(
                gains[v * top_targets + v_slot]);
            const auto adjacency = pair_edge_multiplicity(
                u, v, offsets, neighbors);
            const auto gain = gain_u + gain_v - 2 * adjacency;
            const auto low = min(u, v);
            const auto high = max(u, v);
            if (gain > 0 &&
                (!found || pair_priority_better(
                    gain, low, high, best_gain, min(best_u, best_v),
                    max(best_u, best_v)))) {
                found = true;
                best_gain = gain;
                best_u = u;
                best_v = v;
            }
        }
    }
    pair_u[pair_slot] = best_u;
    pair_v[pair_slot] = best_v;
    pair_result_gains[pair_slot] = found ? best_gain : 0;
    proposed_flags[pair_slot] = found ? 1 : 0;
}

__global__ void select_pair_batch_kernel(
    int pair_slots, const std::int64_t* pair_u, const std::int64_t* pair_v,
    const std::int64_t* pair_result_gains, const std::int64_t* offsets,
    const std::int32_t* neighbors, std::uint8_t* accepted_flags,
    int* accepted_count, std::int64_t* accepted_gain) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int index = 0; index < pair_slots; ++index) accepted_flags[index] = 0;
    int selected = 0;
    std::int64_t gain_sum = 0;
    for (int iteration = 0; iteration < pair_slots; ++iteration) {
        int best = -1;
        for (int index = 0; index < pair_slots; ++index) {
            if (accepted_flags[index] || pair_result_gains[index] <= 0) continue;
            const auto low = min(pair_u[index], pair_v[index]);
            const auto high = max(pair_u[index], pair_v[index]);
            if (best < 0 || pair_priority_better(
                pair_result_gains[index], low, high,
                pair_result_gains[best], min(pair_u[best], pair_v[best]),
                max(pair_u[best], pair_v[best]))) {
                best = index;
            }
        }
        if (best < 0) break;

        bool conflict = false;
        const auto u = pair_u[best];
        const auto v = pair_v[best];
        for (int index = 0; index < pair_slots; ++index) {
            if (!accepted_flags[index]) continue;
            const auto x = pair_u[index];
            const auto y = pair_v[index];
            if (u == x || u == y || v == x || v == y ||
                pair_edge_exists(u, x, offsets, neighbors) ||
                pair_edge_exists(u, y, offsets, neighbors) ||
                pair_edge_exists(v, x, offsets, neighbors) ||
                pair_edge_exists(v, y, offsets, neighbors)) {
                conflict = true;
                break;
            }
        }
        if (!conflict) {
            accepted_flags[best] = 1;
            ++selected;
            gain_sum += pair_result_gains[best];
        }
    }
    *accepted_count = selected;
    *accepted_gain = gain_sum;
}

__global__ void apply_pair_exchange_kernel(
    int pair_slots, const std::uint8_t* accepted_flags,
    const std::int64_t* pair_u, const std::int64_t* pair_v,
    std::int32_t* labels, std::int32_t* old_labels,
    std::uint8_t* moved_flags, std::int64_t* moved_vertices,
    int* moved_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pair_slots || !accepted_flags[index]) return;
    const auto u = pair_u[index];
    const auto v = pair_v[index];
    const int u_source = labels[u];
    const int v_source = labels[v];
    old_labels[u] = u_source;
    old_labels[v] = v_source;
    labels[u] = v_source;
    labels[v] = u_source;
    moved_flags[u] = 1;
    moved_flags[v] = 1;
    const int output = atomicAdd(moved_count, 2);
    moved_vertices[output] = u;
    moved_vertices[output + 1] = v;
}

template <int FixedParts>
__global__ void propose_kernel(
    std::int64_t n, int runtime_parts, int min_gain, bool repair,
    bool balance_mode, bool minimal_balance_repair,
    std::int64_t capacity, std::uint64_t balance_floor,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::int32_t* neighbor_label_counts,
    const std::int32_t* labels, const unsigned long long* loads,
    std::int32_t* targets, std::int32_t* gains) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vertex >= n) return;
    constexpr int storage_parts = FixedParts == 0 ? 32 : FixedParts;
    const int parts = FixedParts == 0 ? runtime_parts : FixedParts;
    const int current = labels[vertex];
    int counts[storage_parts] = {};
    if (neighbor_label_counts) {
        const auto base = static_cast<std::size_t>(vertex) * parts;
        for (int part = 0; part < parts; ++part) {
            counts[part] = neighbor_label_counts[base + part];
        }
    } else {
        for (std::int64_t edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
            const int part = labels[neighbors[edge]];
            if (part >= 0 && part < parts) ++counts[part];
        }
    }
    const bool overloaded = loads[current] > static_cast<unsigned long long>(capacity);
    int best = current;
    int best_gain = 0;
    float best_score = -1.0e30f;
    bool found = false;
    std::uint32_t best_tie = 0;
    for (int part = 0; part < parts; ++part) {
        if (part == current) continue;
        if (loads[part] >= static_cast<unsigned long long>(capacity)) continue;
        const int gain = counts[part] - counts[current];
        float score = static_cast<float>(gain);
        int ranked_gain = gain;
        bool allowed = gain >= min_gain || (repair && overloaded);
        if (balance_mode) {
            if (minimal_balance_repair) {
                score = static_cast<float>(gain);
                allowed = overloaded;
                ranked_gain = gain;
            } else {
                const float target_pressure = fmaxf(
                    0.0f, static_cast<float>(capacity) /
                        static_cast<float>(loads[part] + 1ULL) - 1.0f);
                const float source_pressure = fmaxf(
                    0.0f, static_cast<float>(capacity) /
                        static_cast<float>(max(1ULL, loads[current])) - 1.0f);
                const bool source_high = loads[current] > balance_floor;
                score = static_cast<float>(counts[part]) * target_pressure;
                const float current_score =
                    static_cast<float>(counts[current]) * source_pressure;
                allowed = score > 0.0f &&
                          (overloaded || source_high || score > current_score);
                ranked_gain = min(static_cast<int>(kBalanceGainBound),
                                  max(0, static_cast<int>(score * 1024.0f)));
            }
        }
        if (!allowed) continue;
        const auto tie = hash32(static_cast<std::uint32_t>(vertex) ^
                                static_cast<std::uint32_t>(part * 131));
        const bool better = balance_mode
            ? (!found || score > best_score ||
               (score == best_score && loads[part] < loads[best]) ||
               (score == best_score && loads[part] == loads[best] && tie > best_tie))
            : (!found || gain > best_gain ||
               (gain == best_gain && loads[part] < loads[best]) ||
               (gain == best_gain && loads[part] == loads[best] && tie > best_tie));
        if (better) {
            best = part;
            best_gain = ranked_gain;
            best_score = score;
            best_tie = tie;
            found = true;
        }
    }
    targets[vertex] = best;
    gains[vertex] = best == current ? 0 : best_gain;
}

__global__ void conflict_filter_kernel(
    std::int64_t n, int round, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::int32_t* labels,
    const std::int32_t* targets, const std::int32_t* gains,
    std::int32_t* filtered_targets) {
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
    if (vertex >= n) return;
    const int target = targets[vertex];
    if (target < 0 || target == labels[vertex] || gains[vertex] <= 0) {
        filtered_targets[vertex] = -1;
        return;
    }
    const auto priority = hash32(
        static_cast<std::uint32_t>(vertex) ^
        (static_cast<std::uint32_t>(round) * 0x9e3779b9U));
    for (std::int64_t edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
        const auto neighbor = neighbors[edge];
        if (targets[neighbor] < 0 || targets[neighbor] == labels[neighbor] ||
            gains[neighbor] <= 0) {
            continue;
        }
        const auto neighbor_priority = hash32(
            static_cast<std::uint32_t>(neighbor) ^
            (static_cast<std::uint32_t>(round) * 0x9e3779b9U));
        if (neighbor_priority > priority ||
            (neighbor_priority == priority && neighbor < vertex)) {
            filtered_targets[vertex] = -1;
            return;
        }
    }
    filtered_targets[vertex] = target;
}

__global__ void count_targets_kernel(
    int count, int parts, const std::int64_t* candidates,
    const std::int32_t* targets, unsigned long long* counts) {
    extern __shared__ unsigned long long block_counts[];
    for (int part = threadIdx.x; part < parts; part += blockDim.x) {
        block_counts[part] = 0;
    }
    __syncthreads();
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) atomicAdd(&block_counts[targets[candidates[index]]], 1ULL);
    __syncthreads();
    for (int part = threadIdx.x; part < parts; part += blockDim.x) {
        if (block_counts[part]) atomicAdd(&counts[part], block_counts[part]);
    }
}

__global__ void count_gain_hist_kernel(
    int count, const std::int64_t* candidates, const std::int32_t* targets,
    const std::int32_t* gains, std::int64_t gain_bound,
    unsigned long long* histogram) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto vertex = candidates[index];
    const int target = targets[vertex];
    const std::int64_t range = 2 * gain_bound + 1;
    const std::int64_t raw = static_cast<std::int64_t>(gains[vertex]) + gain_bound;
    const std::int64_t clipped = raw < 0 ? 0 : (raw >= range ? range - 1 : raw);
    const int bin = min(kGainBins - 1,
                        static_cast<int>((clipped * kGainBins) / range));
    atomicAdd(&histogram[static_cast<std::size_t>(target) * kGainBins + bin], 1ULL);
}

__global__ void make_move_keys_kernel(
    int count, const std::int64_t* candidates, const std::int32_t* targets,
    const std::int32_t* gains, std::int64_t gain_bound,
    std::uint64_t* keys) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto vertex = candidates[index];
    const std::int64_t range = 2 * gain_bound + 1;
    const std::int64_t raw = static_cast<std::int64_t>(gains[vertex]) + gain_bound;
    const std::int64_t clipped = raw < 0 ? 0 : (raw >= range ? range - 1 : raw);
    const auto score = static_cast<std::uint32_t>(
        (clipped * 0x07ffffffLL) / range);
    const auto target_key = static_cast<std::uint64_t>(targets[vertex]) << 59;
    const auto score_key = static_cast<std::uint64_t>(0x07ffffffU - score) << 32;
    keys[index] = target_key | score_key |
                  hash32(static_cast<std::uint32_t>(vertex));
}

__device__ __forceinline__ bool reserve_source_slot(
    int source, std::uint32_t* source_quota) {
    auto available = source_quota[source];
    while (available > 0) {
        const auto observed = atomicCAS(
            &source_quota[source], available, available - 1);
        if (observed == available) return true;
        available = observed;
    }
    return false;
}

__global__ void apply_all_kernel(
    int count, const std::int64_t* candidates, const std::int32_t* targets,
    int parts, bool enforce_source_quota, std::uint32_t* source_quota,
    std::int32_t* labels, std::int32_t* old_labels,
    std::uint8_t* moved_flags, unsigned long long* load_deltas, int* changed) {
    extern __shared__ unsigned int block_deltas[];
    const int matrix_size = parts * parts;
    for (int item = threadIdx.x; item < matrix_size; item += blockDim.x) {
        block_deltas[item] = 0;
    }
    __syncthreads();
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const auto vertex = candidates[index];
        const int target = targets[vertex];
        const int source = labels[vertex];
        if (source >= 0 && source < parts && source != target &&
            (!enforce_source_quota || reserve_source_slot(source, source_quota))) {
            atomicAdd(&block_deltas[source * parts + target], 1U);
            old_labels[vertex] = source;
            moved_flags[vertex] = 1;
            labels[vertex] = target;
            atomicAdd(changed, 1);
        }
    }
    __syncthreads();
    for (int item = threadIdx.x; item < matrix_size; item += blockDim.x) {
        if (block_deltas[item]) {
            atomicAdd(&load_deltas[item],
                      static_cast<unsigned long long>(block_deltas[item]));
        }
    }
}

__global__ void apply_quota_kernel(
    int count, const std::int64_t* candidates, const std::int32_t* targets,
    const std::uint32_t* quota, const std::uint32_t* target_begin,
    int parts, bool enforce_source_quota, std::uint32_t* source_quota,
    std::int32_t* labels, std::int32_t* old_labels,
    std::uint8_t* moved_flags, unsigned long long* load_deltas, int* changed) {
    extern __shared__ unsigned int block_deltas[];
    const int matrix_size = parts * parts;
    for (int item = threadIdx.x; item < matrix_size; item += blockDim.x) {
        block_deltas[item] = 0;
    }
    __syncthreads();
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const auto vertex = candidates[index];
        const int target = targets[vertex];
        const auto rank = static_cast<std::uint32_t>(index) - target_begin[target];
        if (rank < quota[target]) {
            const int source = labels[vertex];
            if (source >= 0 && source < parts && source != target &&
                (!enforce_source_quota ||
                 reserve_source_slot(source, source_quota))) {
                atomicAdd(&block_deltas[source * parts + target], 1U);
                old_labels[vertex] = source;
                moved_flags[vertex] = 1;
                labels[vertex] = target;
                atomicAdd(changed, 1);
            }
        }
    }
    __syncthreads();
    for (int item = threadIdx.x; item < matrix_size; item += blockDim.x) {
        if (block_deltas[item]) {
            atomicAdd(&load_deltas[item],
                      static_cast<unsigned long long>(block_deltas[item]));
        }
    }
}

__global__ void count_cut_kernel(
    std::int64_t n, int warp_degree, const std::int64_t* offsets,
    const std::int32_t* neighbors, const std::int32_t* labels,
    unsigned long long* cut) {
    const auto thread = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    std::uint64_t local_cut = 0;
    for (std::int64_t vertex = thread; vertex < n; vertex += stride) {
        if (warp_degree > 0 &&
            offsets[vertex + 1] - offsets[vertex] >= warp_degree) continue;
        const int source = labels[vertex];
        for (std::int64_t edge = offsets[vertex]; edge < offsets[vertex + 1]; ++edge) {
            local_cut += labels[neighbors[edge]] != source;
        }
    }
    if (local_cut) atomicAdd(cut, local_cut);
}

__global__ void count_cut_warp_kernel(
    int vertex_count, const std::int64_t* vertices,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::int32_t* labels, unsigned long long* cut) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= vertex_count) return;
    const auto vertex = vertices[warp];
    const int source = labels[vertex];
    std::uint64_t local_cut = 0;
    for (std::int64_t edge = offsets[vertex] + lane;
         edge < offsets[vertex + 1]; edge += 32) {
        local_cut += labels[neighbors[edge]] != source;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_cut += __shfl_down_sync(0xffffffffu, local_cut, offset);
    }
    if (lane == 0 && local_cut) atomicAdd(cut, local_cut);
}

__global__ void count_cut_delta_warp_kernel(
    int moved_count, const std::int64_t* moved_vertices,
    const std::int64_t* offsets, const std::int32_t* neighbors,
    const std::int32_t* labels, const std::int32_t* old_labels,
    const std::uint8_t* moved_flags, std::int64_t* cut_delta) {
    const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp = global_thread >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= moved_count) return;
    const auto vertex = moved_vertices[warp];
    const int old_source = old_labels[vertex];
    const int new_source = labels[vertex];
    std::int64_t local_delta = 0;
    for (std::int64_t edge = offsets[vertex] + lane;
         edge < offsets[vertex + 1]; edge += 32) {
        const auto neighbor = neighbors[edge];
        const bool neighbor_moved = moved_flags[neighbor] != 0;
        const int old_target = neighbor_moved
            ? old_labels[neighbor]
            : labels[neighbor];
        const int new_target = labels[neighbor];
        const int edge_delta = static_cast<int>(new_source != new_target) -
                               static_cast<int>(old_source != old_target);
        local_delta += static_cast<std::int64_t>(edge_delta) *
                       (neighbor_moved ? 1 : 2);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_delta += __shfl_down_sync(0xffffffffu, local_delta, offset);
    }
    if (lane == 0 && local_delta) {
        atomicAdd(reinterpret_cast<unsigned long long*>(cut_delta),
                  static_cast<unsigned long long>(local_delta));
    }
}

__global__ void sum_candidate_scores_kernel(
    int count, const std::int64_t* candidates, const std::int32_t* gains,
    const std::uint8_t* moved_flags, bool moved_only, std::int64_t* sum) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto vertex = candidates[index];
    if (!moved_only || moved_flags[vertex] != 0) {
        atomicAdd(reinterpret_cast<unsigned long long*>(sum),
                  static_cast<unsigned long long>(gains[vertex]));
    }
}

}

void SingleGPUPartitioner::distance_grow() {
    const int blocks = static_cast<int>((n_ + 255) / 256);
    const cub::CountingInputIterator<std::int64_t> counting(0);
    const std::int64_t init_capacity =
        (n_ + config_.parts - 1) / config_.parts;
    CUDA_CHECK(cudaMemset(d_targets_, 0xff,
                          static_cast<std::size_t>(n_) * sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemset(
        d_signal_, 0, static_cast<std::size_t>(n_) * config_.parts * sizeof(float)));
    for (std::size_t seed_index = 0; seed_index < seeds_.size(); ++seed_index) {
        const int part = static_cast<int>(seed_index % config_.parts);
        distance_bfs(seeds_[seed_index]);
        store_distance_channel_kernel<<<blocks, 256>>>(
            n_, config_.parts, part,
            seed_index >= static_cast<std::size_t>(config_.parts),
            d_labels_, d_signal_);
        CUDA_CHECK(cudaGetLastError());
        const auto seed = seeds_[seed_index];
        CUDA_CHECK(cudaMemcpy(d_targets_ + seed, &part, sizeof(part),
                              cudaMemcpyHostToDevice));
    }

    std::vector<std::uint64_t> loads(
        static_cast<std::size_t>(config_.parts),
        static_cast<std::uint64_t>(config_.seeds_per_part));
    CUDA_CHECK(cudaMemcpy(d_loads_, loads.data(),
                          loads.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    std::uint64_t assigned = static_cast<std::uint64_t>(seeds_.size());
    int rounds = 0;
    while (assigned < static_cast<std::uint64_t>(n_) &&
           rounds < config_.grow_rounds) {
        distance_capacity_propose_kernel<<<blocks, 256>>>(
            n_, config_.parts, d_signal_, d_targets_, d_loads_, init_capacity,
            d_next_labels_, d_gains_);
        CUDA_CHECK(cudaGetLastError());
        const CandidatePredicate predicate{d_next_labels_};
        CUDA_CHECK(cub::DeviceSelect::If(
            d_select_temp_, select_temp_bytes_, counting, d_candidates_,
            d_candidate_count_, static_cast<int>(n_), predicate));
        int candidate_count = 0;
        CUDA_CHECK(cudaMemcpy(&candidate_count, d_candidate_count_, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (candidate_count == 0) break;

        const auto histogram_size = static_cast<std::size_t>(config_.parts) *
                                    kGainBins;
        CUDA_CHECK(cudaMemset(d_gain_hist_, 0,
                              histogram_size * sizeof(unsigned long long)));
        count_distance_hist_kernel<<<(candidate_count + 255) / 256, 256>>>(
            candidate_count, d_candidates_, d_next_labels_, d_gains_,
            d_gain_hist_);
        CUDA_CHECK(cudaGetLastError());
        std::vector<std::uint64_t> histogram(histogram_size, 0);
        CUDA_CHECK(cudaMemcpy(histogram.data(), d_gain_hist_,
                              histogram_size * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost));

        std::vector<std::uint64_t> counts(
            static_cast<std::size_t>(config_.parts), 0);
        for (int part = 0; part < config_.parts; ++part) {
            for (int distance = 0; distance < kGainBins; ++distance) {
                counts[static_cast<std::size_t>(part)] +=
                    histogram[static_cast<std::size_t>(part) * kGainBins + distance];
            }
        }
        std::vector<std::uint32_t> quota(
            static_cast<std::size_t>(config_.parts), 0);
        std::vector<std::uint64_t> accepted(
            static_cast<std::size_t>(config_.parts), 0);
        for (int part = 0; part < config_.parts; ++part) {
            const auto current = loads[static_cast<std::size_t>(part)];
            const std::uint64_t room =
                current >= static_cast<std::uint64_t>(init_capacity)
                    ? 0
                    : static_cast<std::uint64_t>(init_capacity) - current;
            accepted[static_cast<std::size_t>(part)] =
                std::min(counts[static_cast<std::size_t>(part)], room);
            std::uint64_t higher = 0;
            for (int distance = 0;
                 distance < kGainBins &&
                 higher < accepted[static_cast<std::size_t>(part)];
                 ++distance) {
                const auto count = histogram[
                    static_cast<std::size_t>(part) * kGainBins + distance];
                if (higher + count >= accepted[static_cast<std::size_t>(part)]) {
                    quota[static_cast<std::size_t>(part)] =
                        static_cast<std::uint32_t>(
                            higher + accepted[static_cast<std::size_t>(part)] - higher);
                    break;
                }
                higher += count;
            }
        }
        std::vector<std::uint32_t> target_begin(
            static_cast<std::size_t>(config_.parts), 0);
        std::uint64_t prefix = 0;
        for (int part = 0; part < config_.parts; ++part) {
            target_begin[static_cast<std::size_t>(part)] =
                static_cast<std::uint32_t>(prefix);
            prefix += counts[static_cast<std::size_t>(part)];
            loads[static_cast<std::size_t>(part)] +=
                accepted[static_cast<std::size_t>(part)];
        }
        CUDA_CHECK(cudaMemcpy(d_loads_, loads.data(),
                              loads.size() * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_target_quota_, quota.data(),
                              quota.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_target_begin_, target_begin.data(),
                              target_begin.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice));
        make_distance_keys_kernel<<<(candidate_count + 255) / 256, 256>>>(
            candidate_count, d_candidates_, d_next_labels_, d_gains_, d_keys_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_temp_, sort_temp_bytes_, d_keys_, d_sorted_keys_, d_candidates_,
            d_sorted_candidates_, candidate_count, 0, 64));
        CUDA_CHECK(cudaMemset(d_changed_, 0, sizeof(int)));
        apply_initial_quota_kernel<<<(candidate_count + 255) / 256, 256>>>(
            candidate_count, d_sorted_candidates_, d_next_labels_,
            d_target_quota_, d_target_begin_, d_targets_, d_changed_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(d_labels_, d_targets_,
                              static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                              cudaMemcpyDeviceToDevice));
        load_counts(loads);
        assigned = 0;
        for (const auto load : loads) assigned += load;
        ++rounds;
        std::cout << "capacity_distance_round=" << rounds
                  << " assigned=" << assigned << " loads=";
        for (std::size_t i = 0; i < loads.size(); ++i) {
            std::cout << (i ? "," : "") << loads[i];
        }
        std::cout << '\n';
        if (assigned == static_cast<std::uint64_t>(n_)) break;
    }
    if (assigned < static_cast<std::uint64_t>(n_)) {
        std::vector<std::int32_t> host_assignment(static_cast<std::size_t>(n_));
        CUDA_CHECK(cudaMemcpy(host_assignment.data(), d_targets_,
                              host_assignment.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
        std::size_t next_part = 0;
        for (auto& label : host_assignment) {
            if (label >= 0) continue;
            while (next_part < loads.size() &&
                   loads[next_part] >= static_cast<std::uint64_t>(init_capacity)) {
                ++next_part;
            }
            if (next_part == loads.size()) {
                throw std::runtime_error("balanced initialization fallback has no free slot");
            }
            label = static_cast<std::int32_t>(next_part);
            ++loads[next_part];
        }
        CUDA_CHECK(cudaMemcpy(d_targets_, host_assignment.data(),
                              host_assignment.size() * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_labels_, d_targets_,
                          static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                          cudaMemcpyDeviceToDevice));
    std::cout << "capacity_distance assigned_before_fallback=" << assigned
              << " rounds=" << rounds << '\n';
}

void SingleGPUPartitioner::build_current_label_field() {
    const auto start = std::chrono::steady_clock::now();
    const int blocks = static_cast<int>((n_ + 255) / 256);
    const auto elements = static_cast<std::size_t>(n_) * config_.parts;
    CUDA_CHECK(cudaMemset(d_signal_, 0, elements * sizeof(float)));
    label_probe_kernel<<<blocks, 256>>>(
        n_, config_.parts, d_labels_, d_signal_);
    CUDA_CHECK(cudaGetLastError());
    for (int round = 0; round < config_.field_rounds; ++round) {
        switch (config_.parts) {
            case 2:
                structural_diffusion_kernel<2><<<blocks, 256>>>(
                    n_, 2, 0, d_offsets_, d_neighbors_, d_inv_sqrt_degree_,
                    d_signal_, d_next_signal_);
                break;
            case 4:
                structural_diffusion_kernel<4><<<blocks, 256>>>(
                    n_, 4,
                    high_degree_count_ > 0 ? config_.structural_warp_degree : 0,
                    d_offsets_, d_neighbors_, d_inv_sqrt_degree_,
                    d_signal_, d_next_signal_);
                if (high_degree_count_ > 0) {
                    structural_diffusion_warp4_kernel<<<
                        (high_degree_count_ * 32 + 255) / 256, 256>>>(
                        high_degree_count_, d_high_degree_vertices_, d_offsets_,
                        d_neighbors_, d_inv_sqrt_degree_, d_signal_,
                        d_next_signal_);
                }
                break;
            case 8:
                structural_diffusion_kernel<8><<<blocks, 256>>>(
                    n_, 8, 0, d_offsets_, d_neighbors_, d_inv_sqrt_degree_,
                    d_signal_, d_next_signal_);
                break;
            case 16:
                structural_diffusion_kernel<16><<<blocks, 256>>>(
                    n_, 16, 0, d_offsets_, d_neighbors_, d_inv_sqrt_degree_,
                    d_signal_, d_next_signal_);
                break;
            case 32:
                structural_diffusion_kernel<32><<<blocks, 256>>>(
                    n_, 32, 0, d_offsets_, d_neighbors_, d_inv_sqrt_degree_,
                    d_signal_, d_next_signal_);
                break;
            default:
                structural_diffusion_kernel<0><<<blocks, 256>>>(
                    n_, config_.parts, 0, d_offsets_, d_neighbors_,
                    d_inv_sqrt_degree_, d_signal_, d_next_signal_);
                break;
        }
        CUDA_CHECK(cudaGetLastError());
        std::swap(d_signal_, d_next_signal_);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "label_field=current_label_diffusion rounds="
              << config_.field_rounds
              << " seconds=" << seconds << '\n';
}

void SingleGPUPartitioner::build_neighbor_label_counts() {
    if (!config_.cached_neighbor_counts) return;
    const int blocks = static_cast<int>((n_ + 255) / 256);
    switch (config_.parts) {
        case 2:
            build_neighbor_label_counts_kernel<2><<<blocks, 256>>>(
                n_, 2, 0, d_offsets_, d_neighbors_, d_labels_,
                d_neighbor_label_counts_);
            break;
        case 4:
            build_neighbor_label_counts_kernel<4><<<blocks, 256>>>(
                n_, 4,
                high_degree_count_ > 0 ? config_.structural_warp_degree : 0,
                d_offsets_, d_neighbors_, d_labels_,
                d_neighbor_label_counts_);
            if (high_degree_count_ > 0) {
                build_neighbor_label_counts_warp4_kernel<<<
                    (high_degree_count_ * 32 + 255) / 256, 256>>>(
                    high_degree_count_, d_high_degree_vertices_, d_offsets_,
                    d_neighbors_, d_labels_, d_neighbor_label_counts_);
            }
            break;
        case 8:
            build_neighbor_label_counts_kernel<8><<<blocks, 256>>>(
                n_, 8, 0, d_offsets_, d_neighbors_, d_labels_,
                d_neighbor_label_counts_);
            break;
        case 16:
            build_neighbor_label_counts_kernel<16><<<blocks, 256>>>(
                n_, 16, 0, d_offsets_, d_neighbors_, d_labels_,
                d_neighbor_label_counts_);
            break;
        case 32:
            build_neighbor_label_counts_kernel<32><<<blocks, 256>>>(
                n_, 32, 0, d_offsets_, d_neighbors_, d_labels_,
                d_neighbor_label_counts_);
            break;
        default:
            build_neighbor_label_counts_kernel<0><<<blocks, 256>>>(
                n_, config_.parts, 0, d_offsets_, d_neighbors_, d_labels_,
                d_neighbor_label_counts_);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void SingleGPUPartitioner::update_neighbor_label_counts(
    int moved, int candidate_count) {
    if (!config_.cached_neighbor_counts || moved == 0) return;
    const auto delta_limit = static_cast<std::int64_t>(std::floor(
        static_cast<long double>(n_) *
        config_.neighbor_count_delta_max_moved_ratio));
    if (moved > delta_limit) {
        build_neighbor_label_counts();
        return;
    }
    const MovedPredicate moved_predicate{d_moved_flags_};
    CUDA_CHECK(cub::DeviceSelect::If(
        d_select_temp_, select_temp_bytes_, d_candidates_,
        d_sorted_candidates_, d_candidate_count_, candidate_count,
        moved_predicate));
    last_moved_vertices_compacted_ = true;
    update_neighbor_label_counts_warp_kernel<<<
        (moved * 32LL + 255) / 256, 256>>>(
        moved, config_.parts, d_sorted_candidates_, d_offsets_, d_neighbors_,
        d_labels_, d_old_labels_, d_neighbor_label_counts_);
    CUDA_CHECK(cudaGetLastError());
}

void SingleGPUPartitioner::load_counts(std::vector<std::uint64_t>& loads) {
    loads.assign(static_cast<std::size_t>(config_.parts), 0);
    CUDA_CHECK(cudaMemset(d_loads_, 0,
                          loads.size() * sizeof(unsigned long long)));
    const int blocks = static_cast<int>((n_ + 255) / 256);
    count_labels_kernel<<<blocks, 256,
                          static_cast<std::size_t>(config_.parts) *
                              sizeof(unsigned long long)>>>(
        n_, config_.parts, d_labels_, d_loads_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(loads.data(), d_loads_,
                          loads.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
}

std::uint64_t SingleGPUPartitioner::compute_cut() {
    return compute_cut_for_labels(d_labels_);
}

std::uint64_t SingleGPUPartitioner::compute_cut_for_labels(
    const std::int32_t* labels) {
    CUDA_CHECK(cudaMemset(d_cut_, 0, sizeof(unsigned long long)));
    const int blocks = std::min(4096, static_cast<int>((n_ + 255) / 256));
    count_cut_kernel<<<blocks, 256>>>(
        n_, high_degree_count_ > 0 ? config_.structural_warp_degree : 0,
        d_offsets_, d_neighbors_, labels, d_cut_);
    if (high_degree_count_ > 0) {
        count_cut_warp_kernel<<<
            (high_degree_count_ * 32 + 255) / 256, 256>>>(
            high_degree_count_, d_high_degree_vertices_, d_offsets_,
            d_neighbors_, labels, d_cut_);
    }
    CUDA_CHECK(cudaGetLastError());
    std::uint64_t cut = 0;
    CUDA_CHECK(cudaMemcpy(&cut, d_cut_, sizeof(cut), cudaMemcpyDeviceToHost));
    return cut;
}

std::int64_t SingleGPUPartitioner::compute_cut_delta(
    int candidate_count, int moved) {
    if (candidate_count <= 0 || moved <= 0) return 0;
    if (!last_moved_vertices_compacted_) {
        const MovedPredicate moved_predicate{d_moved_flags_};
        CUDA_CHECK(cub::DeviceSelect::If(
            d_select_temp_, select_temp_bytes_, d_candidates_,
            d_sorted_candidates_, d_candidate_count_, candidate_count,
            moved_predicate));
        last_moved_vertices_compacted_ = true;
    }
    CUDA_CHECK(cudaMemset(d_cut_delta_, 0, sizeof(std::int64_t)));
    count_cut_delta_warp_kernel<<<(moved * 32LL + 255) / 256, 256>>>(
        moved, d_sorted_candidates_, d_offsets_, d_neighbors_, d_labels_,
        d_old_labels_, d_moved_flags_, d_cut_delta_);
    CUDA_CHECK(cudaGetLastError());
    std::int64_t delta = 0;
    CUDA_CHECK(cudaMemcpy(&delta, d_cut_delta_, sizeof(delta),
                          cudaMemcpyDeviceToHost));
    return delta;
}

int SingleGPUPartitioner::refine_round_once(
    int round, std::vector<std::uint64_t>& loads,
    bool field_projection, bool balance_projection, bool conflict_aware) {
    last_candidate_count_ = 0;
    last_proposed_candidate_count_ = 0;
    last_quota_count_ = 0;
    last_applied_count_ = 0;
    last_capacity_rejected_ = 0;
    last_candidate_score_sum_ = 0;
    last_applied_score_sum_ = 0;
    last_all_candidates_fit_ = false;
    last_moved_vertices_compacted_ = false;
    const int effective_min_gain =
        field_projection || balance_projection ? 0 : config_.min_gain;
    const std::int64_t effective_gain_bound =
        field_projection ? kFieldGainBound
                         : (balance_projection ? kBalanceGainBound
                                               : gain_abs_bound_);
    const int blocks = static_cast<int>((n_ + 255) / 256);
    const auto balance_floor = static_cast<std::uint64_t>(std::floor(
        static_cast<long double>(active_capacity_) /
        static_cast<long double>(config_.vertex_ratio)));
    if (field_projection) {
        field_target_kernel<<<blocks, 256>>>(
            n_, config_.parts, d_offsets_, d_neighbors_, d_labels_, d_signal_,
            d_targets_, d_gains_);
    } else {
        switch (config_.parts) {
            case 2:
                propose_kernel<2><<<blocks, 256>>>(
                    n_, 2, effective_min_gain, config_.repair,
                    balance_projection, config_.minimal_balance_repair,
                    active_capacity_, balance_floor, d_offsets_, d_neighbors_,
                    config_.cached_neighbor_counts ? d_neighbor_label_counts_ : nullptr,
                    d_labels_, d_loads_, d_targets_, d_gains_);
                break;
            case 4:
                propose_kernel<4><<<blocks, 256>>>(
                    n_, 4, effective_min_gain, config_.repair,
                    balance_projection, config_.minimal_balance_repair,
                    active_capacity_, balance_floor, d_offsets_, d_neighbors_,
                    config_.cached_neighbor_counts ? d_neighbor_label_counts_ : nullptr,
                    d_labels_, d_loads_, d_targets_, d_gains_);
                break;
            case 8:
                propose_kernel<8><<<blocks, 256>>>(
                    n_, 8, effective_min_gain, config_.repair,
                    balance_projection, config_.minimal_balance_repair,
                    active_capacity_, balance_floor, d_offsets_, d_neighbors_,
                    config_.cached_neighbor_counts ? d_neighbor_label_counts_ : nullptr,
                    d_labels_, d_loads_, d_targets_, d_gains_);
                break;
            case 16:
                propose_kernel<16><<<blocks, 256>>>(
                    n_, 16, effective_min_gain, config_.repair,
                    balance_projection, config_.minimal_balance_repair,
                    active_capacity_, balance_floor, d_offsets_, d_neighbors_,
                    config_.cached_neighbor_counts ? d_neighbor_label_counts_ : nullptr,
                    d_labels_, d_loads_, d_targets_, d_gains_);
                break;
            case 32:
                propose_kernel<32><<<blocks, 256>>>(
                    n_, 32, effective_min_gain, config_.repair,
                    balance_projection, config_.minimal_balance_repair,
                    active_capacity_, balance_floor, d_offsets_, d_neighbors_,
                    config_.cached_neighbor_counts ? d_neighbor_label_counts_ : nullptr,
                    d_labels_, d_loads_, d_targets_, d_gains_);
                break;
            default:
                propose_kernel<0><<<blocks, 256>>>(
                    n_, config_.parts, effective_min_gain, config_.repair,
                    balance_projection, config_.minimal_balance_repair,
                    active_capacity_, balance_floor, d_offsets_, d_neighbors_,
                    config_.cached_neighbor_counts ? d_neighbor_label_counts_ : nullptr,
                    d_labels_, d_loads_, d_targets_, d_gains_);
                break;
        }
    }
    CUDA_CHECK(cudaGetLastError());

    const std::int32_t* move_targets = d_targets_;
    if (conflict_aware) {
        conflict_filter_kernel<<<blocks, 256>>>(
            n_, round, d_offsets_, d_neighbors_, d_labels_, d_targets_,
            d_gains_, d_next_labels_);
        CUDA_CHECK(cudaGetLastError());
        move_targets = d_next_labels_;
    }
    const cub::CountingInputIterator<std::int64_t> counting(0);
    const MovePredicate predicate{
        move_targets, d_gains_, d_labels_, d_loads_, active_capacity_,
        effective_min_gain, config_.repair};
    CUDA_CHECK(cub::DeviceSelect::If(
        d_select_temp_, select_temp_bytes_, counting, d_candidates_,
        d_candidate_count_, static_cast<int>(n_), predicate));
    int candidate_count = 0;
    CUDA_CHECK(cudaMemcpy(&candidate_count, d_candidate_count_, sizeof(int),
                          cudaMemcpyDeviceToHost));
    last_candidate_count_ = candidate_count;
    last_proposed_candidate_count_ = candidate_count;
    if (candidate_count == 0) return 0;

    CUDA_CHECK(cudaMemset(d_cut_delta_, 0, sizeof(std::int64_t)));
    sum_candidate_scores_kernel<<<(candidate_count + 255) / 256, 256>>>(
        candidate_count, d_candidates_, d_gains_, d_moved_flags_, false,
        d_cut_delta_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&last_candidate_score_sum_, d_cut_delta_,
                          sizeof(last_candidate_score_sum_),
                          cudaMemcpyDeviceToHost));

    bool all_candidates_fit = false;
    std::vector<std::uint32_t> quota(
        static_cast<std::size_t>(config_.parts), 0);
    std::vector<std::uint32_t> target_begin(
        static_cast<std::size_t>(config_.parts), 0);
    std::vector<std::uint32_t> source_quota(
        static_cast<std::size_t>(config_.parts), 0);
    std::vector<std::uint64_t> counts(
        static_cast<std::size_t>(config_.parts), 0);
    const auto target_limit = static_cast<std::uint64_t>(active_capacity_);

    {
        CUDA_CHECK(cudaMemset(d_loads_, 0,
                              static_cast<std::size_t>(config_.parts) *
                                  sizeof(unsigned long long)));
        count_targets_kernel<<<
            (candidate_count + 255) / 256, 256,
            static_cast<std::size_t>(config_.parts) * sizeof(unsigned long long)>>>(
            candidate_count, config_.parts, d_candidates_, move_targets, d_loads_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(counts.data(), d_loads_,
                              counts.size() * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost));

        all_candidates_fit = true;
        for (int part = 0; part < config_.parts; ++part) {
            if (loads[static_cast<std::size_t>(part)] > target_limit ||
                counts[static_cast<std::size_t>(part)] >
                    target_limit - loads[static_cast<std::size_t>(part)]) {
                all_candidates_fit = false;
                break;
            }
        }
        last_all_candidates_fit_ = all_candidates_fit;

        const auto histogram_size = static_cast<std::size_t>(config_.parts) *
                                    kGainBins;
        CUDA_CHECK(cudaMemset(d_gain_hist_, 0,
                              histogram_size * sizeof(unsigned long long)));
        count_gain_hist_kernel<<<(candidate_count + 255) / 256, 256>>>(
            candidate_count, d_candidates_, move_targets, d_gains_,
            effective_gain_bound, d_gain_hist_);
        CUDA_CHECK(cudaGetLastError());
        std::vector<std::uint64_t> histogram(histogram_size, 0);
        CUDA_CHECK(cudaMemcpy(histogram.data(), d_gain_hist_,
                              histogram_size * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost));

        for (int part = 0; part < config_.parts; ++part) {
            const auto load = loads[static_cast<std::size_t>(part)];
            const auto count = counts[static_cast<std::size_t>(part)];
            if (load >= target_limit || count == 0) continue;
            const std::uint64_t target_quota =
                std::min(target_limit - load, count);
            std::uint64_t higher = 0;
            for (int bin = kGainBins - 1; bin >= 0; --bin) {
                const auto bin_count = histogram[
                    static_cast<std::size_t>(part) * kGainBins + bin];
                if (higher + bin_count >= target_quota) {
                    quota[static_cast<std::size_t>(part)] =
                        static_cast<std::uint32_t>(target_quota);
                    break;
                }
                higher += bin_count;
            }
            if (!field_projection && !balance_projection && !conflict_aware &&
                config_.descent_micro_batch > 0) {
                quota[static_cast<std::size_t>(part)] = std::min(
                    quota[static_cast<std::size_t>(part)],
                    static_cast<std::uint32_t>(config_.descent_micro_batch));
            }
        }
        bool any = false;
        for (const auto value : quota) any = any || value != 0;
        for (const auto value : quota) {
            last_quota_count_ += static_cast<int>(value);
        }
        last_capacity_rejected_ = candidate_count - last_quota_count_;
        if (!any) {
            CUDA_CHECK(cudaMemcpy(d_loads_, loads.data(),
                                  loads.size() * sizeof(std::uint64_t),
                                  cudaMemcpyHostToDevice));
            return 0;
        }
    }

    CUDA_CHECK(cudaMemset(d_changed_, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_moved_flags_, 0,
                          static_cast<std::size_t>(n_) * sizeof(std::uint8_t)));
    const auto delta_elements = static_cast<std::size_t>(config_.parts) *
                                config_.parts;
    CUDA_CHECK(cudaMemset(d_load_deltas_, 0,
                          delta_elements * sizeof(unsigned long long)));
    std::uint64_t prefix = 0;
    for (int part = 0; part < config_.parts; ++part) {
        target_begin[static_cast<std::size_t>(part)] =
            static_cast<std::uint32_t>(prefix);
        prefix += counts[static_cast<std::size_t>(part)];
        if (balance_projection && config_.minimal_balance_repair &&
            loads[static_cast<std::size_t>(part)] > target_limit) {
            source_quota[static_cast<std::size_t>(part)] =
                static_cast<std::uint32_t>(
                    loads[static_cast<std::size_t>(part)] - target_limit);
        }
    }
    const bool enforce_source_quota =
        balance_projection && config_.minimal_balance_repair;
    if (enforce_source_quota) {
        CUDA_CHECK(cudaMemcpy(d_source_quota_, source_quota.data(),
                              source_quota.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice));
    }
    const bool micro_descent =
        !field_projection && !balance_projection && !conflict_aware &&
        config_.descent_micro_batch > 0;
    if (all_candidates_fit && !micro_descent) {
        apply_all_kernel<<<
            (candidate_count + 255) / 256, 256,
            delta_elements * sizeof(unsigned int)>>>(
            candidate_count, d_candidates_, move_targets, config_.parts,
            enforce_source_quota, d_source_quota_, d_labels_, d_old_labels_,
            d_moved_flags_, d_load_deltas_, d_changed_);
        CUDA_CHECK(cudaGetLastError());
    } else {
        CUDA_CHECK(cudaMemcpy(d_target_quota_, quota.data(),
                              quota.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_target_begin_, target_begin.data(),
                              target_begin.size() * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice));
        make_move_keys_kernel<<<(candidate_count + 255) / 256, 256>>>(
            candidate_count, d_candidates_, move_targets, d_gains_,
            effective_gain_bound, d_keys_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            d_sort_temp_, sort_temp_bytes_, d_keys_, d_sorted_keys_,
            d_candidates_, d_sorted_candidates_, candidate_count, 0, 64));
        apply_quota_kernel<<<
            (candidate_count + 255) / 256, 256,
            delta_elements * sizeof(unsigned int)>>>(
            candidate_count, d_sorted_candidates_, move_targets, d_target_quota_,
            d_target_begin_, config_.parts, enforce_source_quota,
            d_source_quota_, d_labels_, d_old_labels_, d_moved_flags_,
            d_load_deltas_, d_changed_);
        CUDA_CHECK(cudaGetLastError());
    }

    std::vector<std::uint64_t> deltas(delta_elements, 0);
    CUDA_CHECK(cudaMemcpy(deltas.data(), d_load_deltas_,
                          delta_elements * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    for (int source = 0; source < config_.parts; ++source) {
        for (int target = 0; target < config_.parts; ++target) {
            if (source == target) continue;
            const auto moved = deltas[static_cast<std::size_t>(source) *
                                      config_.parts + target];
            loads[static_cast<std::size_t>(source)] -= moved;
            loads[static_cast<std::size_t>(target)] += moved;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_loads_, loads.data(),
                          loads.size() * sizeof(std::uint64_t),
                          cudaMemcpyHostToDevice));
    int changed = 0;
    CUDA_CHECK(cudaMemcpy(&changed, d_changed_, sizeof(int),
                          cudaMemcpyDeviceToHost));
    last_applied_count_ = changed;
    CUDA_CHECK(cudaMemset(d_cut_delta_, 0, sizeof(std::int64_t)));
    sum_candidate_scores_kernel<<<(candidate_count + 255) / 256, 256>>>(
        candidate_count, d_candidates_, d_gains_, d_moved_flags_, true,
        d_cut_delta_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&last_applied_score_sum_, d_cut_delta_,
                          sizeof(last_applied_score_sum_),
                          cudaMemcpyDeviceToHost));
    update_neighbor_label_counts(changed, candidate_count);
    return changed;
}

int SingleGPUPartitioner::refine_round(
    int round, std::vector<std::uint64_t>& loads,
    bool field_projection, bool balance_projection, bool conflict_aware) {
    last_refine_cut_delta_ = 0;
    const bool micro_enabled =
        config_.descent_micro_batch > 0 &&
        !field_projection && !balance_projection && !conflict_aware;
    if (!micro_enabled) {
        return refine_round_once(
            round, loads, field_projection, balance_projection,
            conflict_aware);
    }

    // Re-propose after every small submission.  This deliberately changes
    // only ordinary descent: balance, field projection, and polish retain
    // their original batch behavior.
    int total_moved = 0;
    int total_candidates = 0;
    int total_proposed = 0;
    int total_quota = 0;
    int total_applied = 0;
    int total_rejected = 0;
    std::int64_t total_candidate_score = 0;
    std::int64_t total_applied_score = 0;
    std::int64_t total_cut_delta = 0;
    bool all_fit = true;
    for (int micro = 0; micro < config_.descent_micro_rounds; ++micro) {
        const int moved = refine_round_once(
            round, loads, false, false, false);
        if (moved > 0) {
            total_cut_delta += compute_cut_delta(last_candidate_count_, moved);
        }
        total_moved += moved;
        total_candidates += last_candidate_count_;
        total_proposed += last_proposed_candidate_count_;
        total_quota += last_quota_count_;
        total_applied += last_applied_count_;
        total_rejected += last_capacity_rejected_;
        total_candidate_score += last_candidate_score_sum_;
        total_applied_score += last_applied_score_sum_;
        all_fit = all_fit && last_all_candidates_fit_;
        if (moved == 0) break;
    }
    last_candidate_count_ = total_candidates;
    last_proposed_candidate_count_ = total_proposed;
    last_quota_count_ = total_quota;
    last_applied_count_ = total_applied;
    last_capacity_rejected_ = total_rejected;
    last_candidate_score_sum_ = total_candidate_score;
    last_applied_score_sum_ = total_applied_score;
    last_refine_cut_delta_ = total_cut_delta;
    last_all_candidates_fit_ = all_fit;
    // The delta was accumulated before the next micro-batch overwrote the
    // moved flags and candidate list.
    last_moved_vertices_compacted_ = true;
    return total_moved;
}

int SingleGPUPartitioner::pair_exchange_round(
    int round, std::vector<std::uint64_t>& loads,
    std::uint64_t current_cut, std::uint64_t& updated_cut,
    std::uint64_t& proposed_exchanges, std::uint64_t& accepted_exchanges,
    std::int64_t& accepted_gain, int& candidate_records,
    int& boundary_vertices, double& candidate_seconds,
    double& filter_seconds, double& cache_rebuild_seconds,
    double& submit_seconds, double& seconds) {
    const auto start = std::chrono::steady_clock::now();
    updated_cut = current_cut;
    proposed_exchanges = 0;
    accepted_exchanges = 0;
    accepted_gain = 0;
    candidate_records = 0;
    boundary_vertices = 0;
    candidate_seconds = 0.0;
    filter_seconds = 0.0;
    cache_rebuild_seconds = 0.0;
    submit_seconds = 0.0;
    if (config_.pair_verify) {
        const auto verified_before = compute_cut();
        if (verified_before != current_cut) {
            throw std::runtime_error(
                "pair exchange precondition cut verification failed");
        }
    }
    const auto log_pair_round = [&]() {
        seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "  pair_round=" << round
                  << " candidate_records=" << candidate_records
                  << " boundary_vertices=" << boundary_vertices
                  << " proposed_exchanges=" << proposed_exchanges
                  << " accepted_exchanges=" << accepted_exchanges
                  << " conflict_eliminated="
                  << (proposed_exchanges - accepted_exchanges)
                  << " migrated_vertices=" << accepted_exchanges * 2
                  << " accepted_gain=" << accepted_gain
                  << " candidate_seconds=" << candidate_seconds
                  << " filter_seconds=" << filter_seconds
                  << " cache_rebuild_seconds=" << cache_rebuild_seconds
                  << " submit_seconds=" << submit_seconds
                  << " cut_before=" << current_cut
                  << " cut_after=" << updated_cut
                  << " pair_seconds=" << seconds << '\n';
    };

    const int blocks = static_cast<int>((n_ + 255) / 256);
    CUDA_CHECK(cudaMemset(d_pair_boundary_count_, 0, sizeof(int)));
    const auto cached_counts = config_.cached_neighbor_counts
        ? d_neighbor_label_counts_ : nullptr;
    switch (config_.parts) {
        case 2:
            pair_candidate_kernel<2><<<blocks, 256>>>(
                n_, 2, config_.pair_top_targets, d_offsets_, d_neighbors_,
                cached_counts, d_labels_, d_pair_targets_, d_pair_gains_,
                d_pair_boundary_count_);
            break;
        case 4:
            pair_candidate_kernel<4><<<blocks, 256>>>(
                n_, 4, config_.pair_top_targets, d_offsets_, d_neighbors_,
                cached_counts, d_labels_, d_pair_targets_, d_pair_gains_,
                d_pair_boundary_count_);
            break;
        case 8:
            pair_candidate_kernel<8><<<blocks, 256>>>(
                n_, 8, config_.pair_top_targets, d_offsets_, d_neighbors_,
                cached_counts, d_labels_, d_pair_targets_, d_pair_gains_,
                d_pair_boundary_count_);
            break;
        case 16:
            pair_candidate_kernel<16><<<blocks, 256>>>(
                n_, 16, config_.pair_top_targets, d_offsets_, d_neighbors_,
                cached_counts, d_labels_, d_pair_targets_, d_pair_gains_,
                d_pair_boundary_count_);
            break;
        case 32:
            pair_candidate_kernel<32><<<blocks, 256>>>(
                n_, 32, config_.pair_top_targets, d_offsets_, d_neighbors_,
                cached_counts, d_labels_, d_pair_targets_, d_pair_gains_,
                d_pair_boundary_count_);
            break;
        default:
            pair_candidate_kernel<0><<<blocks, 256>>>(
                n_, config_.parts, config_.pair_top_targets, d_offsets_,
                d_neighbors_, cached_counts, d_labels_, d_pair_targets_,
                d_pair_gains_, d_pair_boundary_count_);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&boundary_vertices, d_pair_boundary_count_,
                          sizeof(int), cudaMemcpyDeviceToHost));

    const cub::CountingInputIterator<std::int64_t> counting(0);
    const PairCandidatePredicate predicate{d_pair_targets_,
                                           config_.pair_top_targets};
    CUDA_CHECK(cub::DeviceSelect::If(
        d_select_temp_, select_temp_bytes_, counting, d_pair_values_,
        d_candidate_count_, static_cast<int>(pair_raw_capacity_), predicate));
    int candidate_count = 0;
    CUDA_CHECK(cudaMemcpy(&candidate_count, d_candidate_count_, sizeof(int),
                          cudaMemcpyDeviceToHost));
    candidate_records = candidate_count;
    candidate_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (candidate_count == 0) {
        log_pair_round();
        return 0;
    }

    const auto filter_start = std::chrono::steady_clock::now();
    std::int64_t* current_values = d_pair_values_;
    std::int64_t* next_values = d_pair_sorted_values_;
    for (int stage = 0; stage < 3; ++stage) {
        make_pair_sort_keys_kernel<<<(candidate_count + 255) / 256, 256>>>(
            candidate_count, stage, config_.parts, config_.pair_top_targets,
            current_values, d_pair_targets_, d_pair_gains_, d_labels_,
            d_pair_keys_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
            d_pair_sort_temp_, pair_sort_temp_bytes_, d_pair_keys_,
            d_pair_sorted_keys_, current_values, next_values, candidate_count,
            0, 32));
        std::swap(current_values, next_values);
    }

    CUDA_CHECK(cudaMemset(
        d_pair_bucket_counts_, 0,
        static_cast<std::size_t>(pair_bucket_count_) * sizeof(std::int32_t)));
    count_pair_buckets_kernel<<<(candidate_count + 255) / 256, 256>>>(
        candidate_count, config_.parts, config_.pair_top_targets,
        current_values, d_pair_targets_, d_labels_, d_pair_bucket_counts_);
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::int32_t> bucket_counts(
        static_cast<std::size_t>(pair_bucket_count_), 0);
    CUDA_CHECK(cudaMemcpy(
        bucket_counts.data(), d_pair_bucket_counts_,
        bucket_counts.size() * sizeof(std::int32_t),
        cudaMemcpyDeviceToHost));
    std::vector<std::int32_t> bucket_begin(
        static_cast<std::size_t>(pair_bucket_count_ + 1), 0);
    for (int bucket = 0; bucket < pair_bucket_count_; ++bucket) {
        bucket_begin[static_cast<std::size_t>(bucket + 1)] =
            bucket_begin[static_cast<std::size_t>(bucket)] +
            bucket_counts[static_cast<std::size_t>(bucket)];
    }
    CUDA_CHECK(cudaMemcpy(
        d_pair_bucket_begin_, bucket_begin.data(),
        bucket_begin.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice));

    pair_exchange_kernel<<<(pair_slot_count_ + 255) / 256, 256>>>(
        pair_slot_count_, config_.parts, config_.pair_top_targets,
        config_.pair_bucket_limit, d_pair_bucket_begin_,
        d_pair_bucket_counts_, current_values, d_pair_targets_, d_pair_gains_,
        d_labels_, d_offsets_, d_neighbors_, d_pair_u_, d_pair_v_,
        d_pair_result_gains_, d_pair_proposed_flags_);
    CUDA_CHECK(cudaGetLastError());
    std::vector<std::uint8_t> proposed_flags(
        static_cast<std::size_t>(pair_slot_count_), 0);
    CUDA_CHECK(cudaMemcpy(
        proposed_flags.data(), d_pair_proposed_flags_, proposed_flags.size(),
        cudaMemcpyDeviceToHost));
    for (const auto flag : proposed_flags) proposed_exchanges += flag != 0;

    if (config_.pair_verify) {
        std::vector<std::int32_t> labels_snapshot(
            static_cast<std::size_t>(n_));
        std::vector<std::int64_t> pair_u(
            static_cast<std::size_t>(pair_slot_count_));
        std::vector<std::int64_t> pair_v(
            static_cast<std::size_t>(pair_slot_count_));
        std::vector<std::int64_t> pair_gains(
            static_cast<std::size_t>(pair_slot_count_));
        CUDA_CHECK(cudaMemcpy(labels_snapshot.data(), d_labels_,
                              labels_snapshot.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pair_u.data(), d_pair_u_,
                              pair_u.size() * sizeof(std::int64_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pair_v.data(), d_pair_v_,
                              pair_v.size() * sizeof(std::int64_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pair_gains.data(), d_pair_result_gains_,
                              pair_gains.size() * sizeof(std::int64_t),
                              cudaMemcpyDeviceToHost));
        const auto& offsets = graph_.offsets();
        const auto& neighbors = graph_.neighbors();
        const auto edge_count = [&](std::int64_t source,
                                    std::int64_t target) {
            std::int64_t count = 0;
            for (auto edge = offsets[static_cast<std::size_t>(source)];
                 edge < offsets[static_cast<std::size_t>(source + 1)]; ++edge) {
                count += neighbors[static_cast<std::size_t>(edge)] == target;
            }
            return count;
        };
        const auto move_gain = [&](std::int64_t vertex, int target) {
            const int source = labels_snapshot[static_cast<std::size_t>(vertex)];
            std::int64_t target_count = 0;
            std::int64_t source_count = 0;
            for (auto edge = offsets[static_cast<std::size_t>(vertex)];
                 edge < offsets[static_cast<std::size_t>(vertex + 1)]; ++edge) {
                const int label = labels_snapshot[static_cast<std::size_t>(
                    neighbors[static_cast<std::size_t>(edge)])];
                target_count += label == target;
                source_count += label == source;
            }
            return target_count - source_count;
        };
        for (int index = 0; index < pair_slot_count_; ++index) {
            if (!proposed_flags[static_cast<std::size_t>(index)]) continue;
            const auto u = pair_u[static_cast<std::size_t>(index)];
            const auto v = pair_v[static_cast<std::size_t>(index)];
            const auto exact = move_gain(
                u, labels_snapshot[static_cast<std::size_t>(v)]) +
                move_gain(v, labels_snapshot[static_cast<std::size_t>(u)]) -
                2 * edge_count(u, v);
            if (exact != pair_gains[static_cast<std::size_t>(index)] || exact <= 0) {
                throw std::runtime_error(
                    "pair verification found incorrect proposed gain");
            }
            std::cout << "  pair_proposed round=" << round
                      << " u=" << u << " v=" << v
                      << " gain=" << exact << '\n';
        }
    }

    select_pair_batch_kernel<<<1, 1>>>(
        pair_slot_count_, d_pair_u_, d_pair_v_, d_pair_result_gains_,
        d_offsets_, d_neighbors_, d_pair_accepted_flags_,
        d_pair_accepted_count_, d_pair_accepted_gain_);
    CUDA_CHECK(cudaGetLastError());
    int accepted_count = 0;
    CUDA_CHECK(cudaMemcpy(&accepted_count, d_pair_accepted_count_, sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&accepted_gain, d_pair_accepted_gain_,
                          sizeof(accepted_gain), cudaMemcpyDeviceToHost));
    filter_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - filter_start).count();
    accepted_exchanges = static_cast<std::uint64_t>(accepted_count);
    if (accepted_count == 0) {
        log_pair_round();
        return 0;
    }

    std::vector<std::int32_t> labels_before;
    if (config_.pair_verify) {
        labels_before.resize(static_cast<std::size_t>(n_));
        CUDA_CHECK(cudaMemcpy(labels_before.data(), d_labels_,
                              labels_before.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
    }
    const auto submit_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemset(d_moved_flags_, 0,
                          static_cast<std::size_t>(n_) * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMemset(d_pair_moved_count_, 0, sizeof(int)));
    apply_pair_exchange_kernel<<<(pair_slot_count_ + 255) / 256, 256>>>(
        pair_slot_count_, d_pair_accepted_flags_, d_pair_u_, d_pair_v_,
        d_labels_, d_old_labels_, d_moved_flags_, d_pair_moved_vertices_,
        d_pair_moved_count_);
    CUDA_CHECK(cudaGetLastError());

    int moved_list_count = 0;
    CUDA_CHECK(cudaMemcpy(&moved_list_count, d_pair_moved_count_, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (moved_list_count != accepted_count * 2) {
        throw std::runtime_error("pair exchange migration list count mismatch");
    }

    const auto old_loads = loads;
    load_counts(loads);
    if (loads != old_loads) {
        throw std::runtime_error("pair exchange changed unit vertex loads");
    }
    const auto cache_start = std::chrono::steady_clock::now();
    build_neighbor_label_counts();
    CUDA_CHECK(cudaDeviceSynchronize());
    cache_rebuild_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - cache_start).count();
    updated_cut = compute_cut();
    submit_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - submit_start).count();
    const auto expected_directed_gain =
        static_cast<std::uint64_t>(accepted_gain) * 2ULL;
    if (updated_cut > current_cut ||
        current_cut - updated_cut != expected_directed_gain) {
        throw std::runtime_error(
            "pair exchange directed cut delta disagrees with exact accepted gain");
    }

    if (config_.pair_verify) {
        std::vector<std::int64_t> pair_u(
            static_cast<std::size_t>(pair_slot_count_));
        std::vector<std::int64_t> pair_v(
            static_cast<std::size_t>(pair_slot_count_));
        std::vector<std::int64_t> pair_gains(
            static_cast<std::size_t>(pair_slot_count_));
        std::vector<std::uint8_t> accepted_flags(
            static_cast<std::size_t>(pair_slot_count_));
        CUDA_CHECK(cudaMemcpy(pair_u.data(), d_pair_u_,
                              pair_u.size() * sizeof(std::int64_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pair_v.data(), d_pair_v_,
                              pair_v.size() * sizeof(std::int64_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pair_gains.data(), d_pair_result_gains_,
                              pair_gains.size() * sizeof(std::int64_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(accepted_flags.data(), d_pair_accepted_flags_,
                              accepted_flags.size(), cudaMemcpyDeviceToHost));
        std::vector<std::uint8_t> used(static_cast<std::size_t>(n_), 0);
        std::vector<std::int32_t> labels_after(static_cast<std::size_t>(n_));
        CUDA_CHECK(cudaMemcpy(labels_after.data(), d_labels_,
                              labels_after.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
        const auto& offsets = graph_.offsets();
        const auto& neighbors = graph_.neighbors();
        const auto edge_count = [&](std::int64_t source,
                                    std::int64_t target) {
            std::int64_t count = 0;
            for (auto edge = offsets[static_cast<std::size_t>(source)];
                 edge < offsets[static_cast<std::size_t>(source + 1)]; ++edge) {
                count += neighbors[static_cast<std::size_t>(edge)] == target;
            }
            return count;
        };
        const auto edge_exists = [&](std::int64_t left, std::int64_t right) {
            return edge_count(left, right) > 0 || edge_count(right, left) > 0;
        };
        const auto move_gain = [&](std::int64_t vertex, int target) {
            const int source = labels_before[static_cast<std::size_t>(vertex)];
            std::int64_t target_count = 0;
            std::int64_t source_count = 0;
            for (auto edge = offsets[static_cast<std::size_t>(vertex)];
                 edge < offsets[static_cast<std::size_t>(vertex + 1)]; ++edge) {
                const int label = labels_before[static_cast<std::size_t>(
                    neighbors[static_cast<std::size_t>(edge)])];
                target_count += label == target;
                source_count += label == source;
            }
            return target_count - source_count;
        };
        int checked = 0;
        for (int index = 0; index < pair_slot_count_; ++index) {
            if (!accepted_flags[static_cast<std::size_t>(index)]) continue;
            const auto u = pair_u[static_cast<std::size_t>(index)];
            const auto v = pair_v[static_cast<std::size_t>(index)];
            if (u < 0 || v < 0 || u == v || used[static_cast<std::size_t>(u)] ||
                used[static_cast<std::size_t>(v)]) {
                throw std::runtime_error("pair verification found duplicate endpoint");
            }
            used[static_cast<std::size_t>(u)] = 1;
            used[static_cast<std::size_t>(v)] = 1;
            const int u_target = labels_before[static_cast<std::size_t>(v)];
            const int v_target = labels_before[static_cast<std::size_t>(u)];
            const auto exact = move_gain(u, u_target) + move_gain(v, v_target) -
                               2 * edge_count(u, v);
            if (exact != pair_gains[static_cast<std::size_t>(index)] || exact <= 0) {
                throw std::runtime_error("pair verification found incorrect gain");
            }
            if (labels_after[static_cast<std::size_t>(u)] != u_target ||
                labels_after[static_cast<std::size_t>(v)] != v_target) {
                throw std::runtime_error("pair verification found incorrect labels");
            }
            for (int prior = 0; prior < index; ++prior) {
                if (!accepted_flags[static_cast<std::size_t>(prior)]) continue;
                const auto x = pair_u[static_cast<std::size_t>(prior)];
                const auto y = pair_v[static_cast<std::size_t>(prior)];
                if (u == x || u == y || v == x || v == y ||
                    edge_exists(u, x) || edge_exists(u, y) ||
                    edge_exists(v, x) || edge_exists(v, y)) {
                    throw std::runtime_error(
                        "pair verification found a cross-exchange conflict");
                }
            }
            ++checked;
        }
        if (checked != accepted_count) {
            throw std::runtime_error("pair verification accepted-count mismatch");
        }
        for (const auto label : labels_after) {
            if (label < 0 || label >= config_.parts) {
                throw std::runtime_error("pair verification found invalid label");
            }
        }
    }

    last_candidate_count_ = 0;
    last_moved_vertices_compacted_ = false;
    log_pair_round();
    return accepted_count * 2;
}

bool SingleGPUPartitioner::block_lp_round(
    int round, std::vector<std::uint64_t>& loads,
    std::uint64_t current_cut, std::uint64_t& updated_cut,
    BlockRoundStats& stats) {
    const auto start = std::chrono::steady_clock::now();
    stats = BlockRoundStats{};
    stats.cut_before = current_cut;
    updated_cut = current_cut;
    if (!config_.block_lp || block_candidate_capacity_ == 0) return false;

    const int blocks = static_cast<int>((n_ + 255) / 256);
    const auto cached_counts = config_.cached_neighbor_counts
        ? d_neighbor_label_counts_ : nullptr;
    const auto seed_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemset(
        d_block_seed_vertices_, 0xff,
        static_cast<std::size_t>(block_candidate_capacity_) *
            sizeof(std::int32_t)));
    fill_index_kernel<<<blocks, 256>>>(n_, d_pair_values_);
    CUDA_CHECK(cudaGetLastError());

    std::vector<std::int32_t> bucket_counts(
        static_cast<std::size_t>(pair_bucket_count_), 0);
    std::vector<std::int32_t> bucket_begin(
        static_cast<std::size_t>(pair_bucket_count_ + 1), 0);
    for (int target = 0; target < config_.parts; ++target) {
        std::int64_t* current_values = d_pair_values_;
        std::int64_t* next_values = d_pair_sorted_values_;
        for (int stage = 0; stage < 3; ++stage) {
            block_seed_sort_key_kernel<<<blocks, 256>>>(
                n_, target, config_.parts, stage, current_values,
                d_offsets_, d_neighbors_, cached_counts, d_labels_,
                d_pair_keys_);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
                d_pair_sort_temp_, pair_sort_temp_bytes_, d_pair_keys_,
                d_pair_sorted_keys_, current_values, next_values, n_, 0, 64));
            std::swap(current_values, next_values);
        }

        CUDA_CHECK(cudaMemset(
            d_pair_bucket_counts_, 0,
            static_cast<std::size_t>(pair_bucket_count_) *
                sizeof(std::int32_t)));
        count_block_seed_buckets_kernel<<<blocks, 256>>>(
            n_, target, config_.parts, current_values, d_offsets_,
            d_neighbors_, cached_counts, d_labels_, d_pair_bucket_counts_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(
            bucket_counts.data(), d_pair_bucket_counts_,
            bucket_counts.size() * sizeof(std::int32_t),
            cudaMemcpyDeviceToHost));

        std::fill(bucket_begin.begin(), bucket_begin.end(), 0);
        std::int32_t prefix = 0;
        for (int source = 0; source < config_.parts; ++source) {
            for (int destination = 0; destination < config_.parts;
                 ++destination) {
                const int bucket = source * config_.parts + destination;
                bucket_begin[static_cast<std::size_t>(bucket)] = prefix;
                prefix += bucket_counts[static_cast<std::size_t>(bucket)];
            }
        }
        bucket_begin.back() = prefix;
        CUDA_CHECK(cudaMemcpy(
            d_pair_bucket_begin_, bucket_begin.data(),
            bucket_begin.size() * sizeof(std::int32_t),
            cudaMemcpyHostToDevice));
        const int seed_items = config_.parts * config_.block_seeds_per_pair;
        extract_block_seeds_kernel<<<(seed_items + 255) / 256, 256>>>(
            config_.parts, config_.block_seeds_per_pair, target,
            current_values, d_pair_bucket_counts_, d_pair_bucket_begin_,
            d_offsets_, d_neighbors_, cached_counts, d_labels_,
            d_block_seed_vertices_, d_block_seed_gains_,
            d_block_seed_sources_, d_block_seed_targets_);
        CUDA_CHECK(cudaGetLastError());
        for (int source = 0; source < config_.parts; ++source) {
            if (source != target) {
                stats.seed_pool_candidates += bucket_counts[
                    static_cast<std::size_t>(source) * config_.parts + target];
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.seed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - seed_start).count();

    const auto growth_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemset(
        d_block_candidate_gains_, 0,
        static_cast<std::size_t>(block_candidate_capacity_) *
            sizeof(std::int64_t)));
    CUDA_CHECK(cudaMemset(
        d_block_candidate_sizes_, 0,
        static_cast<std::size_t>(block_candidate_capacity_) *
            sizeof(std::int32_t)));
    CUDA_CHECK(cudaMemset(
        d_block_candidate_valid_, 0,
        static_cast<std::size_t>(block_candidate_capacity_) *
            sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMemset(
        d_block_frontier_overflows_, 0,
        static_cast<std::size_t>(block_candidate_capacity_) *
            sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemset(
        d_block_edge_visits_, 0,
        static_cast<std::size_t>(block_candidate_capacity_) *
            sizeof(std::uint64_t)));
    block_growth_kernel<<<block_candidate_capacity_, 256>>>(
        block_candidate_capacity_, config_.block_max_size,
        config_.block_frontier_limit, d_block_seed_vertices_,
        d_block_seed_gains_, d_block_seed_sources_, d_block_seed_targets_,
        d_offsets_, d_neighbors_, d_labels_, d_block_vertices_,
        d_block_candidate_gains_, d_block_candidate_sizes_,
        d_block_candidate_valid_, d_block_frontier_vertices_,
        d_block_frontier_scores_, d_block_frontier_overflows_,
        d_block_edge_visits_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.growth_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - growth_start).count();

    const int candidate_capacity = block_candidate_capacity_;
    std::vector<std::int32_t> seed_vertices(static_cast<std::size_t>(candidate_capacity));
    std::vector<std::int64_t> seed_gains(static_cast<std::size_t>(candidate_capacity));
    std::vector<std::int32_t> seed_sources(static_cast<std::size_t>(candidate_capacity));
    std::vector<std::int32_t> seed_targets(static_cast<std::size_t>(candidate_capacity));
    std::vector<std::int32_t> candidate_sizes(static_cast<std::size_t>(candidate_capacity));
    std::vector<std::int64_t> candidate_gains(static_cast<std::size_t>(candidate_capacity));
    std::vector<std::uint8_t> candidate_valid(static_cast<std::size_t>(candidate_capacity));
    std::vector<std::int32_t> candidate_vertices(
        static_cast<std::size_t>(candidate_capacity) * config_.block_max_size);
    std::vector<std::uint64_t> frontier_overflows(static_cast<std::size_t>(candidate_capacity));
    std::vector<std::uint64_t> edge_visits(static_cast<std::size_t>(candidate_capacity));
    CUDA_CHECK(cudaMemcpy(seed_vertices.data(), d_block_seed_vertices_,
                          seed_vertices.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seed_gains.data(), d_block_seed_gains_,
                          seed_gains.size() * sizeof(std::int64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seed_sources.data(), d_block_seed_sources_,
                          seed_sources.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(seed_targets.data(), d_block_seed_targets_,
                          seed_targets.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(candidate_sizes.data(), d_block_candidate_sizes_,
                          candidate_sizes.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(candidate_gains.data(), d_block_candidate_gains_,
                          candidate_gains.size() * sizeof(std::int64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(candidate_valid.data(), d_block_candidate_valid_,
                          candidate_valid.size() * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(candidate_vertices.data(), d_block_vertices_,
                          candidate_vertices.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(frontier_overflows.data(), d_block_frontier_overflows_,
                          frontier_overflows.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(edge_visits.data(), d_block_edge_visits_,
                          edge_visits.size() * sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost));

    for (const auto seed : seed_vertices) {
        stats.start_candidates += seed >= 0;
    }

    struct HostCandidate {
        int index = -1;
        int source = -1;
        int target = -1;
        int size = 0;
        std::int64_t gain = 0;
        std::int64_t seed_gain = 0;
    };
    std::vector<HostCandidate> candidates;
    candidates.reserve(static_cast<std::size_t>(candidate_capacity));
    for (int index = 0; index < candidate_capacity; ++index) {
        stats.frontier_overflows += static_cast<int>(frontier_overflows[index]);
        stats.edge_visits += edge_visits[index];
        if (seed_vertices[index] < 0) continue;
        if (candidate_valid[index] == 0 || candidate_sizes[index] <= 0 ||
            candidate_gains[index] <= 0) {
            continue;
        }
        HostCandidate candidate;
        candidate.index = index;
        candidate.source = seed_sources[index];
        candidate.target = seed_targets[index];
        candidate.size = candidate_sizes[index];
        candidate.gain = candidate_gains[index];
        candidate.seed_gain = seed_gains[index];
        candidates.push_back(candidate);
        ++stats.positive_candidates;
        stats.size_ge2_positive += candidate.size >= 2;
        stats.nonpositive_seed_positive += candidate.seed_gain <= 0;
    }
    if (!candidates.empty()) {
        std::vector<std::int64_t> gains;
        std::vector<int> sizes;
        gains.reserve(candidates.size());
        sizes.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            gains.push_back(candidate.gain);
            sizes.push_back(candidate.size);
            if (candidate.gain > stats.best_candidate_gain ||
                (candidate.gain == stats.best_candidate_gain &&
                 (stats.best_candidate_size == 0 ||
                  candidate.size < stats.best_candidate_size))) {
                stats.best_candidate_gain = candidate.gain;
                stats.best_candidate_size = candidate.size;
            }
        }
        std::sort(gains.begin(), gains.end());
        std::sort(sizes.begin(), sizes.end());
        const auto middle = gains.size() / 2;
        stats.median_candidate_gain = gains.size() % 2
            ? static_cast<double>(gains[middle])
            : 0.5 * static_cast<double>(gains[middle - 1] + gains[middle]);
        stats.median_candidate_size = sizes.size() % 2
            ? static_cast<double>(sizes[middle])
            : 0.5 * static_cast<double>(sizes[middle - 1] + sizes[middle]);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const HostCandidate& left, const HostCandidate& right) {
        if (left.gain != right.gain) return left.gain > right.gain;
        if (left.size != right.size) return left.size < right.size;
        if (left.source != right.source) return left.source < right.source;
        if (left.target != right.target) return left.target < right.target;
        return left.index < right.index;
    });
    const auto selection_start = std::chrono::steady_clock::now();
    std::vector<std::uint8_t> used(static_cast<std::size_t>(n_), 0);
    std::vector<std::uint64_t> remaining = loads;
    for (auto& load : remaining) {
        load = load <= static_cast<std::uint64_t>(capacity_)
            ? static_cast<std::uint64_t>(capacity_) - load : 0;
    }
    std::vector<std::uint8_t> selected_flags(
        static_cast<std::size_t>(candidate_capacity), 0);
    for (const auto& candidate : candidates) {
        const auto base = static_cast<std::size_t>(candidate.index) *
                          config_.block_max_size;
        bool overlaps = false;
        for (int item = 0; item < candidate.size; ++item) {
            const auto vertex = candidate_vertices[base + item];
            if (vertex < 0 || vertex >= n_ || used[static_cast<std::size_t>(vertex)]) {
                overlaps = true;
                break;
            }
        }
        if (overlaps) {
            ++stats.overlap_eliminated;
            continue;
        }
        if (remaining[static_cast<std::size_t>(candidate.target)] <
            static_cast<std::uint64_t>(candidate.size)) {
            ++stats.capacity_eliminated;
            continue;
        }
        selected_flags[static_cast<std::size_t>(candidate.index)] = 1;
        ++stats.selected_candidates;
        stats.moved_vertices += candidate.size;
        stats.candidate_gain_sum += candidate.gain;
        remaining[static_cast<std::size_t>(candidate.target)] -=
            static_cast<std::uint64_t>(candidate.size);
        for (int item = 0; item < candidate.size; ++item) {
            const auto vertex = candidate_vertices[base + item];
            used[static_cast<std::size_t>(vertex)] = 1;
        }
    }
    stats.selection_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - selection_start).count();
    if (stats.selected_candidates == 0) {
        stats.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        return false;
    }

    std::vector<std::int32_t> labels_before;
    if (config_.block_verify) {
        labels_before.resize(static_cast<std::size_t>(n_));
        CUDA_CHECK(cudaMemcpy(labels_before.data(), d_labels_,
                              labels_before.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
        std::vector<int> marks(static_cast<std::size_t>(n_), -1);
        for (const auto& candidate : candidates) {
            if (candidate_valid[candidate.index] == 0) continue;
            const auto base = static_cast<std::size_t>(candidate.index) *
                              config_.block_max_size;
            for (int item = 0; item < candidate.size; ++item) {
                const auto vertex = candidate_vertices[base + item];
                if (vertex < 0 || vertex >= n_ ||
                    labels_before[static_cast<std::size_t>(vertex)] !=
                        candidate.source) {
                    throw std::runtime_error(
                        "block verification found an invalid source vertex");
                }
                marks[static_cast<std::size_t>(vertex)] = candidate.index;
            }
            std::int64_t exact_gain = 0;
            for (int item = 0; item < candidate.size; ++item) {
                const auto vertex = candidate_vertices[base + item];
                for (auto edge = graph_.offsets()[static_cast<std::size_t>(vertex)];
                     edge < graph_.offsets()[static_cast<std::size_t>(vertex + 1)];
                     ++edge) {
                    const auto neighbor = graph_.neighbors()[static_cast<std::size_t>(edge)];
                    if (neighbor == vertex) continue;
                    if (labels_before[static_cast<std::size_t>(neighbor)] ==
                        candidate.target) {
                        ++exact_gain;
                    } else if (labels_before[static_cast<std::size_t>(neighbor)] ==
                               candidate.source &&
                               marks[static_cast<std::size_t>(neighbor)] != candidate.index) {
                        --exact_gain;
                    }
                }
            }
            if (exact_gain != candidate.gain) {
                throw std::runtime_error(
                    "block verification found an incorrect candidate gain");
            }
            std::vector<std::int32_t> stack{candidate_vertices[base]};
            std::vector<std::uint8_t> reached(
                static_cast<std::size_t>(candidate.size), 0);
            while (!stack.empty()) {
                const auto vertex = stack.back();
                stack.pop_back();
                for (int item = 0; item < candidate.size; ++item) {
                    if (reached[static_cast<std::size_t>(item)] ||
                        candidate_vertices[base + item] != vertex) continue;
                    reached[static_cast<std::size_t>(item)] = 1;
                    for (auto edge = graph_.offsets()[static_cast<std::size_t>(vertex)];
                         edge < graph_.offsets()[static_cast<std::size_t>(vertex + 1)];
                         ++edge) {
                        const auto neighbor = graph_.neighbors()[static_cast<std::size_t>(edge)];
                        if (marks[static_cast<std::size_t>(neighbor)] == candidate.index) {
                            stack.push_back(neighbor);
                        }
                    }
                    break;
                }
            }
            for (const auto value : reached) {
                if (!value) throw std::runtime_error(
                    "block verification found a disconnected candidate");
            }
        }
    }

    const auto trial_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(d_block_trial_labels_, d_labels_,
                          static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_selected_flags_, selected_flags.data(),
                          selected_flags.size() * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_block_error_, 0, sizeof(int)));
    apply_block_trial_kernel<<<(candidate_capacity + 255) / 256, 256>>>(
        candidate_capacity, config_.block_max_size, d_block_selected_flags_,
        d_block_candidate_sizes_, d_block_seed_sources_, d_block_seed_targets_,
        d_block_vertices_, d_labels_, d_block_trial_labels_, d_block_error_);
    CUDA_CHECK(cudaGetLastError());
    int block_error = 0;
    CUDA_CHECK(cudaMemcpy(&block_error, d_block_error_, sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (block_error) {
        throw std::runtime_error("block trial labels violated source snapshot");
    }
    const auto trial_cut = compute_cut_for_labels(d_block_trial_labels_);
    stats.trial_cut = trial_cut;
    const auto directed_gain = static_cast<std::int64_t>(current_cut) -
                               static_cast<std::int64_t>(trial_cut);
    if ((directed_gain & 1LL) != 0) {
        throw std::runtime_error("block trial cut delta is not symmetric");
    }
    stats.batch_gain = directed_gain / 2;
    stats.trial_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - trial_start).count();

    if (config_.block_verify) {
        std::vector<std::int32_t> trial_labels(static_cast<std::size_t>(n_));
        CUDA_CHECK(cudaMemcpy(trial_labels.data(), d_block_trial_labels_,
                              trial_labels.size() * sizeof(std::int32_t),
                              cudaMemcpyDeviceToHost));
        if (trial_cut != compute_cut_for_labels(d_block_trial_labels_)) {
            throw std::runtime_error("block verification trial cut changed");
        }
        for (const auto& candidate : candidates) {
            if (!selected_flags[static_cast<std::size_t>(candidate.index)]) continue;
            const auto base = static_cast<std::size_t>(candidate.index) *
                              config_.block_max_size;
            for (int item = 0; item < candidate.size; ++item) {
                const auto vertex = candidate_vertices[base + item];
                if (trial_labels[static_cast<std::size_t>(vertex)] !=
                    candidate.target) {
                    throw std::runtime_error(
                        "block verification found an incorrect trial label");
                }
            }
        }
    }

    if (trial_cut >= current_cut) {
        stats.seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        return false;
    }

    CUDA_CHECK(cudaMemcpy(d_labels_, d_block_trial_labels_,
                          static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                          cudaMemcpyDeviceToDevice));
    const auto old_loads = loads;
    load_counts(loads);
    if (loads != old_loads &&
        std::any_of(loads.begin(), loads.end(), [&](std::uint64_t load) {
            return load > static_cast<std::uint64_t>(capacity_);
        })) {
        throw std::runtime_error("block trial exceeded final capacity");
    }
    const auto cache_start = std::chrono::steady_clock::now();
    build_neighbor_label_counts();
    CUDA_CHECK(cudaDeviceSynchronize());
    stats.cache_rebuild_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - cache_start).count();
    updated_cut = trial_cut;
    stats.cut_after = updated_cut;
    stats.accepted = true;
    stats.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    (void)round;
    return true;
}

void SingleGPUPartitioner::run() {
    const auto start = std::chrono::steady_clock::now();
    build_degree_normalization();
    if (initial_labels_ != nullptr) {
        if (initial_labels_->size() != static_cast<std::size_t>(n_)) {
            throw std::runtime_error("in-memory initial partition size mismatch");
        }
        labels_ = *initial_labels_;
        for (const int label : labels_) {
            if (label < 0 || label >= config_.parts) {
                throw std::runtime_error("invalid in-memory initial partition label");
            }
        }
        CUDA_CHECK(cudaMemcpy(d_labels_, labels_.data(),
                              static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                              cudaMemcpyHostToDevice));
    } else if(config_.initial_partition.empty()) {
        choose_distance_separated_seeds();
        distance_grow();
    } else {
        std::ifstream input(config_.initial_partition,std::ios::binary|std::ios::ate);
        if(!input||input.tellg()!=std::streamoff(n_*sizeof(std::int32_t)))throw std::runtime_error("invalid initial partition size");
        input.seekg(0);input.read(reinterpret_cast<char*>(labels_.data()),n_*sizeof(std::int32_t));
        if(!input)throw std::runtime_error("cannot read initial partition");
        for(int label:labels_)if(label<0||label>=config_.parts)throw std::runtime_error("invalid initial partition label");
        CUDA_CHECK(cudaMemcpy(d_labels_,labels_.data(),n_*sizeof(std::int32_t),cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(
        labels_.data(), d_labels_,
        static_cast<std::size_t>(n_) * sizeof(std::int32_t),
        cudaMemcpyDeviceToHost));
    std::uint64_t initial_labels_hash = 1469598103934665603ULL;
    for (const auto label : labels_) {
        initial_labels_hash ^= static_cast<std::uint32_t>(label);
        initial_labels_hash *= 1099511628211ULL;
    }
    build_neighbor_label_counts();

    std::vector<std::uint64_t> loads;
    load_counts(loads);
    const auto imbalance_for = [&](const std::vector<std::uint64_t>& values) {
        const auto max_load = *std::max_element(values.begin(), values.end());
        return n_ ? static_cast<double>(max_load) * config_.parts /
                        static_cast<double>(n_)
                  : 0.0;
    };
    const auto imbalance = [&]() {
        return imbalance_for(loads);
    };
    const auto minimum_ratio = [&]() {
        const auto min_load = *std::min_element(loads.begin(), loads.end());
        return n_ ? static_cast<double>(min_load) * config_.parts /
                        static_cast<double>(n_)
                  : 0.0;
    };
    const auto feasible = [&]() {
        return std::all_of(loads.begin(), loads.end(), [&](std::uint64_t load) {
            return load <= static_cast<std::uint64_t>(capacity_);
        });
    };
    const auto print_load_vector = [&](const std::vector<std::uint64_t>& values) {
        for (std::size_t index = 0; index < values.size(); ++index) {
            std::cout << (index ? "," : "") << values[index];
        }
    };
    const auto print_loads = [&]() { print_load_vector(loads); };

    const std::uint64_t initial_cut = compute_cut();
    std::uint64_t current_cut = initial_cut;
    int incremental_cut_rounds = 0;
    int full_cut_rounds = 0;
    std::uint64_t cycle_field_before_cut = initial_cut;
    std::uint64_t cycle_field_after_cut = initial_cut;
    std::vector<std::uint64_t> cycle_field_before_loads = loads;
    std::vector<std::uint64_t> cycle_field_after_loads = loads;
    bool cycle_field_applied = false;
    std::cout << "phase=initial cut=" << initial_cut
              << " cut_ratio="
              << (m_ ? static_cast<double>(initial_cut) / static_cast<double>(m_)
                     : 0.0)
              << " vertex_imb=" << imbalance()
              << " vertex_min_ratio=" << minimum_ratio()
              << " feasible=" << (feasible() ? 1 : 0)
              << " initial_labels_hash=" << initial_labels_hash
              << " loads=";
    print_loads();
    std::cout << '\n';
    if (config_.enable_field && config_.global_cycles > 0) {
        build_current_label_field();
    }

    CUDA_CHECK(cudaMemcpy(d_best_labels_, d_labels_,
                          static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                          cudaMemcpyDeviceToDevice));
    std::uint64_t best_cut = initial_cut;
    bool have_feasible_best = config_.feasible_recorder && feasible();
    double best_imbalance = imbalance();
    std::string best_source = "initial";
    int best_iteration = -1;

    bool trial_active = false;
    std::uint64_t trial_best_cut = 0;
    bool have_trial_best = false;
    std::string trial_best_source = "none";
    int trial_best_iteration = -1;

    const auto update_cut_after_move = [&](int moved) {
        if (moved == 0) return;
        const auto incremental_limit = static_cast<std::int64_t>(std::floor(
            static_cast<long double>(n_) *
            config_.incremental_cut_max_moved_ratio));
        if (config_.incremental_cut && moved <= incremental_limit) {
            const auto delta = compute_cut_delta(last_candidate_count_, moved);
            const auto updated = static_cast<std::int64_t>(current_cut) + delta;
            if (updated < 0) {
                throw std::runtime_error("incremental cut became negative");
            }
            current_cut = static_cast<std::uint64_t>(updated);
            ++incremental_cut_rounds;
            if (config_.incremental_cut_verify) {
                const auto verified = compute_cut();
                if (verified != current_cut) {
                    throw std::runtime_error(
                        "incremental cut verification failed: expected " +
                        std::to_string(verified) + ", got " +
                        std::to_string(current_cut));
                }
            }
        } else {
            current_cut = compute_cut();
            ++full_cut_rounds;
        }
    };

    auto run_operator = [&](int iteration, const char* name,
                            bool field_projection, bool balance_projection,
                            bool conflict_aware = false) {
        const auto operator_start = std::chrono::steady_clock::now();
        const auto before = current_cut;
        const int moved = refine_round(
            iteration, loads, field_projection, balance_projection,
            conflict_aware);
        const bool micro_descent = config_.descent_micro_batch > 0 &&
                                   !field_projection &&
                                   !balance_projection && !conflict_aware;
        if (micro_descent && moved > 0) {
            const auto updated = static_cast<std::int64_t>(current_cut) +
                                 last_refine_cut_delta_;
            if (updated < 0) {
                throw std::runtime_error("micro-batch incremental cut became negative");
            }
            current_cut = static_cast<std::uint64_t>(updated);
            ++incremental_cut_rounds;
            if (config_.incremental_cut_verify) {
                const auto verified = compute_cut();
                if (verified != current_cut) {
                    throw std::runtime_error(
                        "micro-batch incremental cut verification failed");
                }
            }
        } else {
            update_cut_after_move(moved);
        }
        const auto operator_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - operator_start).count();
        std::cout << "  " << name << "=" << iteration
                  << " moved=" << moved
                  << " candidates=" << last_candidate_count_
                  << " quota=" << last_quota_count_
                  << " capacity_rejected=" << last_capacity_rejected_
                  << " applied=" << last_applied_count_
                  << " all_fit=" << (last_all_candidates_fit_ ? 1 : 0)
                  << " score_sum=" << last_candidate_score_sum_
                  << " applied_score_sum=" << last_applied_score_sum_
                  << " cut_before=" << before
                  << " cut_after=" << current_cut
                  << " directed_delta="
                  << (static_cast<std::int64_t>(before) -
                      static_cast<std::int64_t>(current_cut))
                  << " interaction_delta="
                  << (static_cast<std::int64_t>(before) -
                      static_cast<std::int64_t>(current_cut) -
                      2 * last_applied_score_sum_)
                  << " vertex_imb=" << imbalance()
                  << " vertex_min_ratio=" << minimum_ratio()
                  << " operator_seconds=" << operator_seconds << '\n';
        return moved;
    };

    auto evaluate_stage = [&](const char* name, int iteration,
                              std::int64_t moved) {
        const auto candidate_cut = current_cut;
        const bool state_feasible = feasible();
        const auto best_before = have_feasible_best ? best_cut : 0;
        const auto trial_before = have_trial_best ? trial_best_cut : 0;
        bool global_checkpointed = false;
        bool trial_checkpointed = false;
        if (config_.feasible_recorder && state_feasible &&
            (!have_feasible_best || candidate_cut < best_cut)) {
            best_cut = candidate_cut;
            have_feasible_best = true;
            best_imbalance = imbalance();
            best_source = name;
            best_iteration = iteration;
            global_checkpointed = true;
            CUDA_CHECK(cudaMemcpy(
                d_best_labels_, d_labels_,
                static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                cudaMemcpyDeviceToDevice));
        }
        if (config_.feasible_recorder && trial_active && state_feasible &&
            (!have_trial_best || candidate_cut < trial_best_cut)) {
            trial_best_cut = candidate_cut;
            have_trial_best = true;
            trial_best_source = name;
            trial_best_iteration = iteration;
            trial_checkpointed = true;
        }
        const char* action = !config_.feasible_recorder
            ? "disabled"
            : (global_checkpointed && trial_checkpointed
                   ? "global+trial_checkpoint"
                   : (global_checkpointed
                          ? "global_checkpoint"
                          : (trial_checkpointed ? "trial_checkpoint" : "keep")));
        std::cout << "  stage=" << name << iteration
                  << " candidate_cut=" << candidate_cut
                  << " cut=" << candidate_cut
                  << " cut_ratio="
                  << (m_ ? static_cast<double>(candidate_cut) /
                               static_cast<double>(m_)
                         : 0.0)
                  << " feasible=" << (state_feasible ? 1 : 0)
                  << " vertex_imb=" << imbalance()
                  << " vertex_min_ratio=" << minimum_ratio()
                  << " loads=";
        print_loads();
        std::cout << " moved=" << moved
                  << " best_before=" << best_before
                  << " best_after=" << (have_feasible_best ? best_cut : 0)
                  << " global_best_source=" << best_source
                  << " global_best_iteration=" << best_iteration
                  << " trial_best_before=" << trial_before
                  << " trial_best_after="
                  << (have_trial_best ? trial_best_cut : 0)
                  << " trial_best_source=" << trial_best_source
                  << " trial_best_iteration=" << trial_best_iteration
                  << " action=" << action << '\n';
    };

    int iteration = 0;
    if (config_.enable_field && config_.global_cycles > 0) {
        active_capacity_ = exploration_capacity_;
        cycle_field_before_cut = current_cut;
        cycle_field_before_loads = loads;
        const int field_iteration = iteration++;
        const int moved = run_operator(field_iteration, "field", true, false);
        evaluate_stage("field", field_iteration, moved);
        cycle_field_applied = true;
        cycle_field_after_cut = current_cut;
        cycle_field_after_loads = loads;
    }

    const int descent_rounds = config_.refine_rounds;
    bool time_budget_stopped = false;
    int completed_global_cycles = 0;
    double budget_stop_elapsed = 0.0;
    for (int cycle = 0; cycle < config_.global_cycles; ++cycle) {
        if (cycle > 0 || !config_.enable_field) {
            cycle_field_applied = false;
            cycle_field_before_cut = current_cut;
            cycle_field_after_cut = current_cut;
            cycle_field_before_loads = loads;
            cycle_field_after_loads = loads;
        }
        const auto anneal_capacity = static_cast<std::int64_t>(std::floor(
            (static_cast<long double>(n_) / config_.parts) *
            (config_.vertex_ratio + 0.05L)));
        active_capacity_ = cycle == 0
            ? exploration_capacity_
            : (cycle == 1 ? anneal_capacity : capacity_);
        const bool checkpoint_for_budget = config_.time_budget_seconds > 0.0;
        const auto cycle_start_cut = current_cut;
        const auto cycle_start_loads = loads;
        const auto cycle_start_best_cut = best_cut;
        const auto cycle_start_have_best = have_feasible_best;
        const auto cycle_start_best_imbalance = best_imbalance;
        const auto cycle_start_best_source = best_source;
        const auto cycle_start_best_iteration = best_iteration;
        const auto cycle_start_iteration = iteration;
        if (checkpoint_for_budget) {
            CUDA_CHECK(cudaMemcpy(
                d_time_checkpoint_labels_, d_labels_,
                static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(
                d_time_checkpoint_best_labels_, d_best_labels_,
                static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                cudaMemcpyDeviceToDevice));
        }
        if (cycle > 0) {
            if (config_.restore_best_cycle && have_feasible_best &&
                (!feasible() || current_cut > best_cut)) {
                CUDA_CHECK(cudaMemcpy(d_labels_, d_best_labels_,
                    static_cast<std::size_t>(n_) * sizeof(std::int32_t), cudaMemcpyDeviceToDevice));
                load_counts(loads);
                build_neighbor_label_counts();
                current_cut = best_cut;
                std::cout << "cycle_restore=" << cycle << " cut=" << best_cut << '\n';
            }
            if (config_.enable_field) {
                cycle_field_before_cut = current_cut;
                cycle_field_before_loads = loads;
                build_current_label_field();
                const int field_iteration = iteration++;
                const int moved = run_operator(
                    field_iteration, "field", true, false);
                evaluate_stage("field", field_iteration, moved);
                cycle_field_applied = true;
                cycle_field_after_cut = current_cut;
                cycle_field_after_loads = loads;
            }
        }

        const auto anchor_cut = current_cut;
        const bool anchor_feasible = feasible();
        const double anchor_imbalance = imbalance();
        trial_active = true;
        trial_best_cut = anchor_cut;
        have_trial_best = anchor_feasible;
        trial_best_source = anchor_feasible
            ? (config_.enable_field ? "field" : "anchor")
            : "none";
        trial_best_iteration = anchor_feasible && config_.enable_field
            ? iteration - 1 : -1;

        std::int64_t balance_moved = 0;
        for (int round = 0; round < config_.balance_rounds; ++round) {
            const int balance_iteration = iteration++;
            const int moved = run_operator(
                balance_iteration, "balance", false, true);
            balance_moved += moved;
            evaluate_stage("balance", balance_iteration, moved);
            if (moved == 0) break;
        }
        const auto balance_cut = current_cut;
        const bool balance_feasible = feasible();
        const double balance_imbalance = imbalance();

        std::int64_t descent_moved = 0;
        std::uint64_t cycle_cuts[2] = {current_cut, 0};
        std::vector<std::uint64_t> cycle_loads[2];
        cycle_loads[0] = loads;
        int cycle_states = 1;
        int cycle_confirmations = 0;
        for (int round = 0; round < descent_rounds; ++round) {
            const int descent_iteration = iteration++;
            const int moved = run_operator(
                descent_iteration, "descent", false, false);
            descent_moved += moved;
            evaluate_stage("descent", descent_iteration, moved);
            if (moved == 0) break;
            const int cycle_slot = (round + 1) & 1;
            const bool repeats_two_back =
                config_.oscillation_guard && cycle_states >= 2 &&
                current_cut == cycle_cuts[cycle_slot] &&
                loads == cycle_loads[cycle_slot];
            cycle_confirmations = repeats_two_back
                ? cycle_confirmations + 1
                : 0;
            cycle_cuts[cycle_slot] = current_cut;
            cycle_loads[cycle_slot] = loads;
            ++cycle_states;
            if (cycle_confirmations >= 2) {
                std::cout << "  descent_stop=two_cycle round=" << round
                          << " cut=" << current_cut << '\n';
                break;
            }
        }
        const auto descent_cut = current_cut;
        const bool descent_feasible = feasible();
        const double descent_imbalance = imbalance();
        std::int64_t pair_moved = 0;
        std::int64_t pair_descent_moved = 0;
        double pair_descent_seconds = 0.0;
        if (config_.pair_exchange && config_.pair_exchange_rounds > 0 &&
            feasible()) {
            active_capacity_ = capacity_;
            for (int pair_round = 0;
                 pair_round < config_.pair_exchange_rounds; ++pair_round) {
                std::uint64_t updated_cut = current_cut;
                std::uint64_t proposed_exchanges = 0;
                std::uint64_t accepted_exchanges = 0;
                std::int64_t accepted_gain = 0;
                int candidate_records = 0;
                int boundary_vertices = 0;
                double candidate_seconds = 0.0;
                double filter_seconds = 0.0;
                double cache_rebuild_seconds = 0.0;
                double submit_seconds = 0.0;
                double pair_seconds = 0.0;
                const int moved = pair_exchange_round(
                    pair_round, loads, current_cut, updated_cut,
                    proposed_exchanges, accepted_exchanges, accepted_gain,
                    candidate_records, boundary_vertices, candidate_seconds,
                    filter_seconds, cache_rebuild_seconds, submit_seconds,
                    pair_seconds);
                current_cut = updated_cut;
                std::cout << "  pair_summary=" << pair_round
                          << " candidate_records=" << candidate_records
                          << " boundary_vertices=" << boundary_vertices
                          << " proposed_exchanges=" << proposed_exchanges
                          << " accepted_exchanges=" << accepted_exchanges
                          << " conflict_eliminated="
                          << (proposed_exchanges - accepted_exchanges)
                          << " migrated_vertices=" << accepted_exchanges * 2
                          << " accepted_gain=" << accepted_gain
                          << " candidate_seconds=" << candidate_seconds
                          << " filter_seconds=" << filter_seconds
                          << " cache_rebuild_seconds="
                          << cache_rebuild_seconds
                          << " submit_seconds=" << submit_seconds
                          << " pair_seconds=" << pair_seconds << '\n';
                if (moved == 0) break;
                pair_moved += accepted_exchanges;
                evaluate_stage("pair", iteration++, moved);

                const auto pair_descent_start = std::chrono::steady_clock::now();
                for (int descent_round = 0;
                     descent_round < descent_rounds; ++descent_round) {
                    const int pair_descent_iteration = iteration++;
                    const int descent_move = run_operator(
                        pair_descent_iteration, "pair_descent", false, false);
                    pair_descent_moved += descent_move;
                    evaluate_stage(
                        "pair_descent", pair_descent_iteration, descent_move);
                    if (descent_move == 0) break;
                }
                pair_descent_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - pair_descent_start).count();
                if (!feasible()) {
                    throw std::runtime_error(
                        "pair exchange descent violated final capacity");
                }
            }
        }
        const auto post_pair_cut = current_cut;
        const bool post_pair_feasible = feasible();
        const double post_pair_imbalance = imbalance();
        const auto trial_best_seen = have_trial_best ? trial_best_cut : 0;
        trial_active = false;
        const auto balance_damage = static_cast<std::int64_t>(balance_cut) -
                                    static_cast<std::int64_t>(anchor_cut);
        const auto descent_recovery = static_cast<std::int64_t>(balance_cut) -
                                      static_cast<std::int64_t>(descent_cut);
        std::cout << "trial=" << cycle
                  << " anchor_cut=" << anchor_cut
                  << " balance_cut=" << balance_cut
                  << " descent_cut=" << descent_cut
                  << " pair_cut=" << post_pair_cut
                  << " trial_cut=" << post_pair_cut
                  << " trial_best_seen_cut=" << trial_best_seen
                  << " balance_moved=" << balance_moved
                  << " descent_moved=" << descent_moved
                  << " pair_moved=" << pair_moved
                  << " pair_descent_moved=" << pair_descent_moved
                  << " pair_descent_seconds=" << pair_descent_seconds
                  << " balance_damage=" << balance_damage
                  << " descent_recovery=" << descent_recovery
                  << " pair_gain="
                  << (static_cast<std::int64_t>(descent_cut) -
                      static_cast<std::int64_t>(post_pair_cut))
                  << " net_gain="
                  << (static_cast<std::int64_t>(anchor_cut) -
                      static_cast<std::int64_t>(post_pair_cut))
                  << " anchor_feasible=" << (anchor_feasible ? 1 : 0)
                  << " balance_feasible=" << (balance_feasible ? 1 : 0)
                  << " descent_feasible=" << (descent_feasible ? 1 : 0)
                  << " pair_feasible=" << (post_pair_feasible ? 1 : 0)
                  << " trial_feasible=" << (post_pair_feasible ? 1 : 0)
                  << " trial_best_source=" << trial_best_source
                  << " trial_best_iteration=" << trial_best_iteration
                  << " anchor_imb=" << anchor_imbalance
                  << " balance_imb=" << balance_imbalance
                  << " descent_imb=" << descent_imbalance
                  << " trial_imb=" << descent_imbalance
                  << " best_feasible_cut=" << best_cut
                  << " best_feasible_imb=" << best_imbalance
                  << " field_applied=" << (cycle_field_applied ? 1 : 0)
                  << " field_before_cut=" << cycle_field_before_cut
                  << " field_after_cut=" << cycle_field_after_cut
                  << " field_before_imb="
                  << imbalance_for(cycle_field_before_loads)
                  << " field_after_imb="
                  << imbalance_for(cycle_field_after_loads)
                  << " field_before_loads=";
        print_load_vector(cycle_field_before_loads);
        std::cout << " field_after_loads=";
        print_load_vector(cycle_field_after_loads);
        std::cout << " elapsed_seconds="
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start).count()
                  << '\n';

        ++completed_global_cycles;
        if (checkpoint_for_budget) {
            const auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > config_.time_budget_seconds) {
                time_budget_stopped = true;
                budget_stop_elapsed = elapsed;
                CUDA_CHECK(cudaMemcpy(
                    d_labels_, d_time_checkpoint_labels_,
                    static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(
                    d_best_labels_, d_time_checkpoint_best_labels_,
                    static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToDevice));
                load_counts(loads);
                loads = cycle_start_loads;
                build_neighbor_label_counts();
                current_cut = cycle_start_cut;
                best_cut = cycle_start_best_cut;
                have_feasible_best = cycle_start_have_best;
                best_imbalance = cycle_start_best_imbalance;
                best_source = cycle_start_best_source;
                best_iteration = cycle_start_best_iteration;
                iteration = cycle_start_iteration;
                active_capacity_ = capacity_;
                completed_global_cycles = cycle;
                std::cout << "time_budget_stop attempted_cycle=" << cycle
                          << " completed_global_cycles=" << completed_global_cycles
                          << " elapsed_seconds=" << elapsed
                          << " budget_seconds=" << config_.time_budget_seconds
                          << " overshoot_seconds="
                          << (elapsed - config_.time_budget_seconds)
                          << " restored_to_cycle_start=1\n";
                break;
            }
        }
    }
    active_capacity_ = capacity_;

    if (config_.feasible_recorder && !have_feasible_best) {
        throw std::runtime_error(
            "search produced no partition satisfying the maximum load bound");
    }
    auto run_polish = [&](const char* stage_name) {
        const auto polish_start = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaMemcpy(
            d_labels_, d_best_labels_,
            static_cast<std::size_t>(n_) * sizeof(std::int32_t),
            cudaMemcpyDeviceToDevice));
        load_counts(loads);
        build_neighbor_label_counts();
        current_cut = best_cut;
        const auto polish_before = best_cut;
        for (int round = 0; round < config_.polish_rounds; ++round) {
            const int polish_iteration = iteration++;
            const int moved = refine_round(
                polish_iteration, loads, false, false, true);
            update_cut_after_move(moved);
            const bool state_feasible = feasible();
            std::cout << "  " << stage_name << "=" << round
                      << " moved=" << moved
                      << " cut=" << current_cut
                      << " feasible=" << (state_feasible ? 1 : 0)
                      << " vertex_imb=" << imbalance()
                      << " vertex_min_ratio=" << minimum_ratio() << '\n';
            const bool improved = state_feasible && current_cut < best_cut;
            if (improved) {
                best_cut = current_cut;
                best_imbalance = imbalance();
                best_source = stage_name;
                best_iteration = polish_iteration;
                CUDA_CHECK(cudaMemcpy(
                    d_best_labels_, d_labels_,
                    static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToDevice));
            }
            if (moved == 0 || !improved) break;
        }
        CUDA_CHECK(cudaMemcpy(
            d_labels_, d_best_labels_,
            static_cast<std::size_t>(n_) * sizeof(std::int32_t),
            cudaMemcpyDeviceToDevice));
        load_counts(loads);
        const auto polish_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - polish_start).count();
        std::cout << "polish_summary=" << stage_name
                  << " before_cut=" << polish_before
                  << " after_cut=" << best_cut
                  << " gain="
                  << (static_cast<std::int64_t>(polish_before) -
                      static_cast<std::int64_t>(best_cut)) / 2.0
                  << " seconds=" << polish_seconds << '\n';
        return static_cast<double>(
            static_cast<std::int64_t>(polish_before) -
            static_cast<std::int64_t>(best_cut)) / 2.0;
    };

    if (config_.feasible_recorder && have_feasible_best) {
        run_polish("polish");
    }

    if (config_.block_lp && config_.block_rounds > 0 &&
        config_.feasible_recorder && have_feasible_best) {
        active_capacity_ = capacity_;
        for (int block_round = 0;
             block_round < config_.block_rounds; ++block_round) {
            BlockRoundStats stats;
            std::uint64_t updated_cut = current_cut;
            const bool accepted = block_lp_round(
                block_round, loads, current_cut, updated_cut, stats);
            current_cut = updated_cut;
            if (accepted) {
                if (current_cut >= best_cut) {
                    throw std::runtime_error(
                        "accepted block update did not improve best cut");
                }
                best_cut = current_cut;
                best_imbalance = imbalance();
                best_source = "block";
                best_iteration = iteration++;
                CUDA_CHECK(cudaMemcpy(
                    d_best_labels_, d_labels_,
                    static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                    cudaMemcpyDeviceToDevice));
                stats.post_polish_gain = run_polish("block_polish");
            }
            stats.cut_after = accepted ? updated_cut : current_cut;
            std::cout << "  block_round=" << block_round
                      << " start_candidates=" << stats.start_candidates
                      << " seed_pool_candidates=" << stats.seed_pool_candidates
                      << " positive_candidates=" << stats.positive_candidates
                      << " size_ge2_positive=" << stats.size_ge2_positive
                      << " nonpositive_seed_positive="
                      << stats.nonpositive_seed_positive
                      << " frontier_overflows=" << stats.frontier_overflows
                      << " overlap_eliminated=" << stats.overlap_eliminated
                      << " capacity_eliminated=" << stats.capacity_eliminated
                      << " selected_candidates=" << stats.selected_candidates
                      << " moved_vertices=" << stats.moved_vertices
                      << " candidate_gain_sum=" << stats.candidate_gain_sum
                      << " best_candidate_gain=" << stats.best_candidate_gain
                      << " median_candidate_gain=" << stats.median_candidate_gain
                      << " best_candidate_size=" << stats.best_candidate_size
                      << " median_candidate_size=" << stats.median_candidate_size
                      << " batch_gain=" << stats.batch_gain
                      << " post_polish_gain=" << stats.post_polish_gain
                      << " edge_visits=" << stats.edge_visits
                      << " cut_before=" << stats.cut_before
                      << " trial_cut=" << stats.trial_cut
                      << " cut_after=" << stats.cut_after
                      << " accepted=" << (stats.accepted ? 1 : 0)
                      << " seed_seconds=" << stats.seed_seconds
                      << " growth_seconds=" << stats.growth_seconds
                      << " selection_seconds=" << stats.selection_seconds
                      << " trial_seconds=" << stats.trial_seconds
                      << " cache_rebuild_seconds="
                      << stats.cache_rebuild_seconds
                      << " block_seconds=" << stats.seconds << '\n';
            if (!accepted) break;
        }
    }
    if (config_.feasible_recorder && have_feasible_best) {
        std::cout << "best_cut=" << best_cut << '\n';
        std::cout << "best_feasible_source=" << best_source
                  << " best_feasible_iteration=" << best_iteration << '\n';
    }
    CUDA_CHECK(cudaMemcpy(labels_.data(), d_labels_,
                          static_cast<std::size_t>(n_) * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost));
    initialized_ = true;
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "cut_updates incremental_rounds=" << incremental_cut_rounds
              << " full_rounds=" << full_cut_rounds << '\n';
    std::cout << "time_budget_summary enabled="
              << (config_.time_budget_seconds > 0.0 ? 1 : 0)
              << " stopped=" << (time_budget_stopped ? 1 : 0)
              << " budget_seconds=" << config_.time_budget_seconds
              << " completed_global_cycles=" << completed_global_cycles
              << " stop_elapsed_seconds=" << budget_stop_elapsed
              << " actual_seconds=" << seconds
              << " deviation_seconds="
              << (config_.time_budget_seconds > 0.0
                      ? seconds - config_.time_budget_seconds : 0.0)
              << '\n';
    std::cout << "partition_seconds=" << seconds << '\n';
    const auto result = metrics();
    const auto max_load = *std::max_element(
        result.vertex_loads.begin(), result.vertex_loads.end());
    const auto min_load = *std::min_element(
        result.vertex_loads.begin(), result.vertex_loads.end());
    std::cout << "final_cut=" << result.cut
              << " edge_entries=" << result.edges
              << " cut_ratio="
              << (result.edges ? static_cast<double>(result.cut) /
                                     static_cast<double>(result.edges)
                               : 0.0)
              << " vertex_imb="
              << (n_ ? static_cast<double>(max_load) * config_.parts /
                            static_cast<double>(n_)
                     : 0.0)
              << " vertex_min_ratio="
              << (n_ ? static_cast<double>(min_load) * config_.parts /
                            static_cast<double>(n_)
                     : 0.0)
              << " feasible="
              << (max_load <= static_cast<std::uint64_t>(capacity_)
                      ? 1
                      : 0)
              << " loads=";
    for (std::size_t index = 0; index < result.vertex_loads.size(); ++index) {
        std::cout << (index ? "," : "") << result.vertex_loads[index];
    }
    std::cout << '\n';
}

SingleGPUMetrics SingleGPUPartitioner::metrics() const {
    if (!initialized_) throw std::runtime_error("partitioner has not run");
    SingleGPUMetrics result;
    result.edges = static_cast<std::uint64_t>(m_);
    result.vertex_loads.assign(static_cast<std::size_t>(config_.parts), 0);
    for (const auto label : labels_) {
        if (label >= 0 && label < config_.parts) {
            ++result.vertex_loads[static_cast<std::size_t>(label)];
        }
    }
    auto* self = const_cast<SingleGPUPartitioner*>(this);
    result.cut = self->compute_cut();
    return result;
}

void SingleGPUPartitioner::save(const std::string& output_path) {
    if (!initialized_) throw std::runtime_error("partitioner has not run");
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open " + output_path);
    output.write(reinterpret_cast<const char*>(labels_.data()),
                 static_cast<std::streamsize>(
                     labels_.size() * sizeof(std::int32_t)));
    if (!output) throw std::runtime_error("cannot write " + output_path);
}
