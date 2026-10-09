#pragma once

#include "graph.hpp"
#include "hierarchy.hpp"

#include <cstdint>
#include <vector>

namespace gpart {

struct RefineOptions {
    int parts = 4;
    double imbalance_ratio = 1.03;
    int max_rounds = 32;
    bool strict_verify = false;
    int stale_rounds = 3;
    double refinement_growth_ratio = 2.0;
};

struct RefineStats {
    int rounds = 0;
    std::uint64_t proposals = 0;
    std::uint64_t accepted = 0;
    std::uint64_t initial_cut = 0;
    std::uint64_t final_cut = 0;
    std::uint64_t final_max_part_weight = 0;
    double seconds = 0.0;
    int admission_iterations = 0;
    int admission_fallbacks = 0;
};

struct RefineLevelTimings {
    double workspace_resize_seconds = 0.0;
    double graph_h2d_seconds = 0.0;
    double projection_gpu_seconds = 0.0;
    double refinement_seconds = 0.0;
    double verification_seconds = 0.0;
};

struct RefineLevelResult {
    std::size_t level = 0;
    std::int64_t vertices = 0;
    std::int64_t edge_entries = 0;
    std::uint64_t projection_cut = 0;
    std::uint64_t projection_max_part_weight = 0;
    double projection_imbalance = 0.0;
    RefineStats refinement;
    RefineLevelTimings timings;
};

template <typename Types>
struct DeviceUncoarsenResult {
    std::vector<typename Types::VertexT> partition;
    thrust::device_vector<typename Types::VertexT> device_partition;
    std::vector<RefineLevelResult> levels;
    std::vector<std::uint64_t> final_part_weights;
    std::uint64_t final_cut = 0;
    std::uint64_t total_edge_weight = 0;
    double final_d2h_seconds = 0.0;
};

template <typename Types>
void refine_partition(
    const WeightedGraph<Types>& graph,
    std::vector<typename Types::VertexT>& partition,
    const RefineOptions& options,
    RefineStats* stats = nullptr);

template <typename Types>
DeviceUncoarsenResult<Types> refine_hierarchy_device(
    const Hierarchy<Types>& hierarchy,
    const std::vector<typename Types::VertexT>& coarsest_partition,
    const RefineOptions& options);

template <typename Types>
DeviceUncoarsenResult<Types> refine_hierarchy_device(
    DeviceHierarchy<Types>& hierarchy,
    thrust::device_vector<typename Types::VertexT>&& coarsest_partition,
    const RefineOptions& options);

}  // namespace gpart
