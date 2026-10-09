#include "refine.hpp"

#include "check.hpp"
#include "detail/scratch_pool.hpp"
#include "graph_types.hpp"
#include "partition_capacity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cub/block/block_scan.cuh>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>

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

// Rebalancing proposals are grouped by overloaded source and loss per unit weight.
struct RebalanceProposal {
    std::uint32_t source;
    double gain_per_weight;
};

template <typename OffsetT, typename WeightT, typename VertexT>
__global__ void refine_cut_kernel(
    std::int64_t n,
    const OffsetT* offsets,
    const VertexT* neighbors,
    const WeightT* edge_weights,
    const VertexT* partition,
    unsigned long long* directed_cut) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const auto vertex = static_cast<std::int64_t>(blockIdx.x) *
        kWarpsPerBlock + warp;
    __shared__ unsigned long long warp_cuts[kWarpsPerBlock];
    unsigned long long local = 0;
    if (vertex < n) {
        const auto source = partition[vertex];
        for (std::int64_t edge = static_cast<std::int64_t>(offsets[vertex]) + lane;
             edge < static_cast<std::int64_t>(offsets[vertex + 1]); edge += kWarpSize) {
            if (source != partition[neighbors[edge]]) {
                local += static_cast<unsigned long long>(edge_weights[edge]);
            }
        }
    }
    // Sum the same directed entries as before, using exact uint64 arithmetic.
    // Tail warps participate with zero so every thread reaches the barrier.
    for (int delta = kWarpSize / 2; delta; delta /= 2)
        local += __shfl_down_sync(0xffffffffU, local, delta);
    if (lane == 0) warp_cuts[warp] = local;
    __syncthreads();
    if (warp == 0) {
        local = lane < kWarpsPerBlock ? warp_cuts[lane] : 0;
        for (int delta = kWarpSize / 2; delta; delta /= 2)
            local += __shfl_down_sync(0xffffffffU, local, delta);
        if (lane == 0 && local != 0) atomicAdd(directed_cut, local);
    }
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
    if (n == 0) return 0;
    const int blocks = static_cast<int>((n + kWarpsPerBlock - 1) / kWarpsPerBlock);
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
    return partition_capacity(total_weight, parts, imbalance_ratio);
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
class RefineWorkspace {
public:
    using VertexT = typename Types::VertexT;
    detail::ScratchVector<VertexT> previous_partition;
    detail::ScratchVector<unsigned long long> previous_part_weights;
    thrust::device_vector<unsigned long long> cut_counter;
    detail::ScratchVector<RebalanceProposal> rebalance_proposals;
    detail::ScratchVector<std::uint64_t> dense_affinity;
    detail::ScratchVector<std::int32_t> incremental_targets;
    detail::ScratchVector<std::int64_t> incremental_gains;
    detail::ScratchVector<std::int32_t> incremental_locks;
    detail::ScratchVector<std::int32_t> incremental_flags;
    detail::ScratchVector<VertexT> incremental_movers;
    detail::ScratchVector<std::int64_t> incremental_delta;
    detail::ScratchVector<std::int32_t> balance_bucket_ids;
    detail::ScratchVector<std::uint64_t> balance_bucket_ranks;
    detail::ScratchVector<unsigned long long> balance_bucket_counts;
    detail::ScratchVector<unsigned long long> balance_bucket_offsets;
    double resize_seconds = 0.0;

    template <typename T, typename Allocator>
    void grow(thrust::device_vector<T, Allocator>& buffer, std::size_t required) {
        if (buffer.size() < required) {
            const auto start = std::chrono::steady_clock::now();
            buffer.resize(required);
            resize_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        }
    }

    void initialize(std::size_t vertices, std::size_t parts) {
        grow(previous_partition, vertices);
        grow(previous_part_weights, parts);
        grow(cut_counter, 1);
        grow(rebalance_proposals, vertices);
    }
};

template <typename Types>
class RefineDeviceContext {
public:
    using VertexT = typename Types::VertexT;
    DeviceWeightedGraph<Types> graph;
    thrust::device_vector<VertexT> partition;
    thrust::device_vector<unsigned long long> part_weights;
    RefineWorkspace<Types>& workspace;
    std::vector<std::uint64_t> host_part_weights;
    std::uint64_t capacity;
    std::int64_t n;
    int vertex_blocks;
    int warp_blocks;

    RefineDeviceContext(
        const WeightedGraph<Types>& input,
        const std::vector<VertexT>& labels,
        const RefineOptions& options,
        RefineWorkspace<Types>& temporary)
        : workspace(temporary),
          n(input.vertices()),
          vertex_blocks(static_cast<int>((n + 255) / 256)),
          warp_blocks(static_cast<int>(
              (n + kWarpsPerBlock - 1) / kWarpsPerBlock)) {
        if (options.parts < 2 || options.parts > 32 ||
            options.max_rounds < 0 || options.max_rounds > 64 ||
            options.stale_rounds < 1 || options.stale_rounds > 64 ||
            !std::isfinite(options.refinement_growth_ratio) || options.refinement_growth_ratio < 1.0 ||
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
        DeviceWeightedGraph<Types>&& device_graph,
        thrust::device_vector<VertexT>&& device_partition,
        thrust::device_vector<unsigned long long>&& device_part_weights,
        std::vector<std::uint64_t> measured_part_weights,
        const RefineOptions& options,
        RefineWorkspace<Types>& temporary)
        : graph(std::move(device_graph)),
          partition(std::move(device_partition)),
          part_weights(std::move(device_part_weights)),
          workspace(temporary),
          host_part_weights(std::move(measured_part_weights)),
          n(graph.vertices()),
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
        workspace.initialize(static_cast<std::size_t>(n), part_weights.size());
    }

    std::uint64_t cut() {
        return device_cut_value(graph, partition, workspace.cut_counter);
    }

    void sync_part_weights() {
        thrust::copy(
            part_weights.begin(), part_weights.end(),
            host_part_weights.begin());
    }

    void export_partition(std::vector<VertexT>& labels) {
        thrust::copy(partition.begin(), partition.end(), labels.begin());
    }
};

template <typename Types>
void validate_refinement_gain_range(const WeightedGraph<Types>& graph) {
    constexpr auto signed_limit = static_cast<std::uint64_t>(
        std::numeric_limits<std::int64_t>::max()) / 2;
    constexpr auto max_weight = static_cast<std::uint64_t>(
        std::numeric_limits<typename Types::WeightT>::max());
    // A type/size bound proves exactly the old safety condition without
    // touching edge data. Division avoids overflow in the bound itself.
    if (graph.edge_weights.size() <= signed_limit / max_weight) return;
    std::uint64_t directed_weight = 0;
    for (const auto edge_weight : graph.edge_weights) {
        const auto weight = static_cast<std::uint64_t>(edge_weight);
        if (weight > signed_limit ||
            directed_weight > signed_limit - weight) {
            throw std::overflow_error(
                "refinement signed gain range exceeded");
        }
        directed_weight += weight;
    }
}

struct RefinementWeightSum {
    __host__ __device__ std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        constexpr auto limit = std::uint64_t(INT64_MAX) / 2;
        return a > limit || b > limit || a > limit - b ? limit + 1 : a + b;
    }
};

template <typename Types>
void validate_refinement_gain_range(const DeviceWeightedGraph<Types>& graph) {
    constexpr auto limit = std::uint64_t(INT64_MAX) / 2;
    constexpr auto max_weight = std::numeric_limits<typename Types::WeightT>::max();
    if (graph.edge_weights.size() <= limit / max_weight) return;
    if (thrust::reduce(graph.edge_weights.begin(), graph.edge_weights.end(),
                       std::uint64_t{0}, RefinementWeightSum{}) > limit) {
        throw std::overflow_error("refinement signed gain range exceeded");
    }
}

struct InvalidId {
    std::int64_t bound;
    template <typename T>
    __host__ __device__ bool operator()(T id) const { return id < 0 || id >= bound; }
};

struct CheckedWeight {
    std::uint64_t value;
    bool overflow;
    __host__ __device__ CheckedWeight(std::uint64_t v = 0) : value(v), overflow(false) {}
};
struct CheckedWeightAdd {
    __host__ __device__ CheckedWeight operator()(CheckedWeight a, CheckedWeight b) const {
        CheckedWeight out(a.value + b.value);
        out.overflow = a.overflow || b.overflow || out.value < a.value;
        return out;
    }
};
struct ToCheckedWeight {
    template <typename W>
    __host__ __device__ CheckedWeight operator()(W weight) const { return CheckedWeight(weight); }
};

#include "refine_incremental.cuh"

template <typename Types>
void refine_partition(
    const WeightedGraph<Types>& graph,
    std::vector<typename Types::VertexT>& partition,
    const RefineOptions& options,
    RefineStats* output_stats) {
    RefineWorkspace<Types> workspace;
    RefineDeviceContext<Types> context(graph, partition, options, workspace);
    auto stats = run_incremental_refinement(context, options, options.max_rounds);
    context.export_partition(partition);
    if (output_stats != nullptr) *output_stats = stats;
}

template <typename Types, bool Resident, typename H, typename P>
DeviceUncoarsenResult<Types> refine_hierarchy_impl(
    H& hierarchy, P&& coarsest_partition,
    const RefineOptions& options) {
    using VertexT = typename Types::VertexT;
    if (options.parts < 2 || options.parts > 32 || options.max_rounds < 0 ||
        options.max_rounds > 64 || options.stale_rounds < 1 || options.stale_rounds > 64 ||
        !std::isfinite(options.refinement_growth_ratio) || options.refinement_growth_ratio < 1.0 ||
        !std::isfinite(options.imbalance_ratio) ||
        options.imbalance_ratio < 1.0) {
        throw std::invalid_argument("invalid refinement options");
    }
    if (hierarchy.levels.empty() ||
        hierarchy.fine_to_coarse.size() + 1 != hierarchy.levels.size()) {
        throw std::invalid_argument("resident refinement requires a complete hierarchy");
    }
    // Validate once before any refinement, including externally built
    // hierarchies; do not assume their coarse edge weights are bounded by
    // the original graph's weight sum.
    for (std::size_t level = 0; level < hierarchy.levels.size(); ++level) {
        validate_refinement_gain_range(hierarchy.levels[level]);
    }
    RefineWorkspace<Types> workspace;
    auto& coarsest = hierarchy.levels.back();
    std::uint64_t total_weight = 0;
    const auto initial_h2d_start = std::chrono::steady_clock::now();
    DeviceWeightedGraph<Types> current_graph;
    thrust::device_vector<VertexT> current_partition;
    thrust::device_vector<unsigned long long> current_part_weights;
    std::vector<std::uint64_t> current_host_weights;
    if constexpr (Resident) {
        if (coarsest_partition.size() != static_cast<std::size_t>(coarsest.vertices()) ||
            thrust::count_if(coarsest_partition.begin(), coarsest_partition.end(),
                             InvalidId{options.parts}) != 0) {
            throw std::invalid_argument("invalid coarsest device partition");
        }
        const auto sum = thrust::transform_reduce(coarsest.vertex_weights.begin(),
            coarsest.vertex_weights.end(), ToCheckedWeight{}, CheckedWeight{}, CheckedWeightAdd{});
        if (sum.overflow) throw std::overflow_error("refinement vertex weight sum overflow");
        total_weight = sum.value;
        current_graph = std::move(coarsest);
        current_partition = std::move(coarsest_partition);
        device_part_weights(current_graph, current_partition, current_part_weights, options.parts);
        current_host_weights.resize(options.parts);
        thrust::copy(current_part_weights.begin(), current_part_weights.end(), current_host_weights.begin());
    } else {
        current_host_weights = validate_and_measure_partition(
            coarsest, coarsest_partition, options.parts, total_weight);
        current_graph = make_device_weighted(coarsest);
        current_partition = coarsest_partition;
        current_part_weights.assign(current_host_weights.begin(), current_host_weights.end());
    }
    if (*std::max_element(current_host_weights.begin(), current_host_weights.end()) >
        refinement_capacity(total_weight, options.parts, options.imbalance_ratio)) {
        throw std::invalid_argument("initial refinement partition is imbalanced");
    }
    double pending_graph_h2d = Resident ? 0.0 : std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initial_h2d_start).count();
    workspace.grow(workspace.cut_counter, 1);
    auto& cut_counter = workspace.cut_counter;

    DeviceUncoarsenResult<Types> result;
    result.levels.reserve(hierarchy.levels.size() - 1);
    std::uint64_t last_cut=0;
    auto last_refined_vertices=current_graph.vertices();
    {
        RefineDeviceContext<Types> context(std::move(current_graph),
            std::move(current_partition), std::move(current_part_weights),
            current_host_weights, options, workspace);
        const auto stats = run_incremental_refinement(context, options, options.max_rounds);
        last_cut=stats.final_cut;
        std::cout << "refine_coarsest vertices=" << context.n
                  << " initial_cut=" << stats.initial_cut
                  << " final_cut=" << stats.final_cut
                  << " rounds=" << stats.rounds
                  << " seconds=" << stats.seconds << '\n';
        current_graph = std::move(context.graph);
        current_partition = std::move(context.partition);
        current_part_weights = std::move(context.part_weights);
        current_host_weights = context.host_part_weights;
    }

    for (std::size_t coarse_level = hierarchy.levels.size() - 1;
         coarse_level > 0; --coarse_level) {
        const auto fine_level = coarse_level - 1;
        auto& fine = hierarchy.levels[fine_level];
        auto& map = hierarchy.fine_to_coarse[fine_level];
        if (map.size() != static_cast<std::size_t>(fine.vertices())) {
            throw std::invalid_argument("fine-to-coarse map has wrong length");
        }
        if constexpr (Resident) {
            if (thrust::count_if(map.begin(), map.end(), InvalidId{current_graph.vertices()}) != 0) {
                throw std::invalid_argument("fine-to-coarse map contains an invalid id");
            }
        } else {
            for (const auto coarse_vertex : map) {
                if (coarse_vertex < 0 || coarse_vertex >= current_graph.vertices()) {
                    throw std::invalid_argument("fine-to-coarse map contains an invalid id");
                }
            }
        }

        const auto workspace_start = result.levels.empty()
            ? 0.0 : workspace.resize_seconds;
        RefineLevelResult report;
        report.level = fine_level;
        report.vertices = fine.vertices();
        report.edge_entries = fine.edges();
        const auto graph_h2d_start = std::chrono::steady_clock::now();
        DeviceWeightedGraph<Types> fine_graph;
        thrust::device_vector<VertexT> device_map;
        if constexpr (Resident) {
            fine_graph = std::move(fine);
            device_map = std::move(map);
        } else {
            fine_graph = make_device_weighted(fine);
            device_map = map;
        }
        thrust::device_vector<VertexT> fine_partition(
            static_cast<std::size_t>(report.vertices));
        report.timings.graph_h2d_seconds = Resident ? 0.0 : pending_graph_h2d +
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - graph_h2d_start).count();
        pending_graph_h2d = 0.0;

        const auto projection_start = std::chrono::steady_clock::now();
        const int projection_blocks = static_cast<int>(
            (report.vertices + 255) / 256);
        project_partition_kernel<<<projection_blocks, 256>>>(
            report.vertices, thrust::raw_pointer_cast(device_map.data()),
            thrust::raw_pointer_cast(current_partition.data()),
            thrust::raw_pointer_cast(fine_partition.data()));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        report.timings.projection_gpu_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - projection_start).count();

        thrust::device_vector<unsigned long long> fine_part_weights;
        auto projected_weights = current_host_weights;
        if (options.strict_verify) {
            const auto verification_start = std::chrono::steady_clock::now();
            device_part_weights(
                fine_graph, fine_partition, fine_part_weights, options.parts);
            thrust::copy(
                fine_part_weights.begin(), fine_part_weights.end(),
                projected_weights.begin());
            report.projection_cut = device_cut_value(
                fine_graph, fine_partition, cut_counter);
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
            report.timings.verification_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - verification_start).count();
        } else {
            // Projection preserves part weights; capacity needs no rescan.
            fine_part_weights = std::move(current_part_weights);
        }
        report.projection_max_part_weight = *std::max_element(
            projected_weights.begin(), projected_weights.end());
        const auto projected_total = std::accumulate(
            projected_weights.begin(), projected_weights.end(),
            std::uint64_t{0});
        report.projection_imbalance = projected_total == 0 ? 0.0 :
            static_cast<double>(report.projection_max_part_weight) *
            options.parts / static_cast<double>(projected_total);

        RefineDeviceContext<Types> context(
            std::move(fine_graph), std::move(fine_partition),
            std::move(fine_part_weights), projected_weights, options, workspace);
        auto resize_before_stage = workspace.resize_seconds;
        const bool refine_this_level = fine_level==0 ||
            report.vertices>=options.refinement_growth_ratio*last_refined_vertices;
        if (refine_this_level) {
            report.refinement = run_incremental_refinement(context, options, options.max_rounds);
            last_refined_vertices=report.vertices;
        } else {
            // Projection preserves cut and part weights. Avoid rebuilding
            // the affinity cache when this level adds little resolution.
            report.refinement.initial_cut=report.refinement.final_cut=last_cut;
            report.refinement.final_max_part_weight=report.projection_max_part_weight;
        }
        if (!options.strict_verify) {
            report.projection_cut = report.refinement.initial_cut;
        }
        report.timings.refinement_seconds = report.refinement.seconds -
            (workspace.resize_seconds - resize_before_stage);
        report.timings.workspace_resize_seconds = workspace.resize_seconds - workspace_start;
        last_cut = report.refinement.final_cut;
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
    if constexpr (Resident) {
        result.device_partition = std::move(current_partition);
    } else {
        result.partition.resize(static_cast<std::size_t>(current_graph.vertices()));
        const auto final_d2h_start = std::chrono::steady_clock::now();
        thrust::copy(current_partition.begin(), current_partition.end(), result.partition.begin());
        result.final_d2h_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - final_d2h_start).count();
    }
    return result;
}

template <typename Types>
DeviceUncoarsenResult<Types> refine_hierarchy_device(
    const Hierarchy<Types>& hierarchy,
    const std::vector<typename Types::VertexT>& partition, const RefineOptions& options) {
    return refine_hierarchy_impl<Types, false>(hierarchy, partition, options);
}

template <typename Types>
DeviceUncoarsenResult<Types> refine_hierarchy_device(
    DeviceHierarchy<Types>& hierarchy,
    thrust::device_vector<typename Types::VertexT>&& partition, const RefineOptions& options) {
    return refine_hierarchy_impl<Types, true>(hierarchy, std::move(partition), options);
}

template void refine_partition<ActiveTypes>(
    const WeightedGraph<ActiveTypes>&,
    std::vector<ActiveTypes::VertexT>&,
    const RefineOptions&,
    RefineStats*);

template DeviceUncoarsenResult<ActiveTypes>
refine_hierarchy_device<ActiveTypes>(
    const Hierarchy<ActiveTypes>&,
    const std::vector<ActiveTypes::VertexT>&,
    const RefineOptions&);
template DeviceUncoarsenResult<ActiveTypes> refine_hierarchy_device<ActiveTypes>(
    DeviceHierarchy<ActiveTypes>&, thrust::device_vector<ActiveTypes::VertexT>&&,
    const RefineOptions&);

}  // namespace gpart
