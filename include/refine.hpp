#pragma once

#include "graph.hpp"
#include "hierarchy.hpp"

#include <cstdint>
#include <vector>

namespace gpart {

struct RefineOptions {
    int parts = 4;
    double imbalance_ratio = 1.10;
    std::uint32_t seed = 0;
    int max_rounds = 4;
    bool strict_verify = false;
};

struct RefineStats {
    int rounds = 0;
    std::uint64_t proposals = 0;
    std::uint64_t accepted = 0;
    std::uint64_t initial_cut = 0;
    std::uint64_t final_cut = 0;
    std::uint64_t final_max_part_weight = 0;
    double seconds = 0.0;
};

struct PairRefineStats {
    std::uint64_t candidates = 0;
    std::uint64_t mutual_pairs = 0;
    std::uint64_t accepted = 0;
    std::uint64_t cut_before = 0;
    std::uint64_t cut_after = 0;
    bool rollback = false;
};

struct RefineLevelTimings {
    double workspace_resize_seconds = 0.0;
    double graph_h2d_seconds = 0.0;
    double projection_gpu_seconds = 0.0;
    double plain_seconds = 0.0;
    double pair_seconds = 0.0;
    double cleanup_seconds = 0.0;
    double verification_seconds = 0.0;
};

struct RefineLevelResult {
    std::size_t level = 0;
    std::int64_t vertices = 0;
    std::int64_t edge_entries = 0;
    std::uint64_t projection_cut = 0;
    std::uint64_t projection_max_part_weight = 0;
    double projection_imbalance = 0.0;
    RefineStats plain;
    PairRefineStats pair;
    RefineStats cleanup;
    RefineLevelTimings timings;
};

template <typename Types>
struct DeviceUncoarsenResult {
    std::vector<typename Types::VertexT> partition;
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
void coordinated_pair_escape(
    const WeightedGraph<Types>& graph,
    std::vector<typename Types::VertexT>& partition,
    const RefineOptions& options,
    int level,
    PairRefineStats* stats = nullptr);

template <typename Types>
DeviceUncoarsenResult<Types> refine_hierarchy_device(
    const Hierarchy<Types>& hierarchy,
    const std::vector<typename Types::VertexT>& coarsest_partition,
    const RefineOptions& options);

}  // namespace gpart
