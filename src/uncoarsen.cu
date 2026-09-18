#include "uncoarsen.hpp"

#include "graph_types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gpart {
namespace {

std::uint64_t sum_weights(const std::vector<std::uint64_t>& weights) {
    std::uint64_t total = 0;
    for (const auto weight : weights) {
        if (total > std::numeric_limits<std::uint64_t>::max() - weight) {
            throw std::overflow_error("total vertex weight overflow");
        }
        total += weight;
    }
    return total;
}

double imbalance_of(
    const std::vector<std::uint64_t>& weights, std::uint64_t total) {
    if (total == 0) return 0.0;
    const auto maximum = *std::max_element(weights.begin(), weights.end());
    return static_cast<double>(maximum) * weights.size() /
        static_cast<double>(total);
}

}  // namespace

template <typename Types, typename H, typename P>
DeviceUncoarsenResult<Types> uncoarsen_impl(
    H& hierarchy, P&& coarsest_partition,
    const RefineOptions& options) {
    const auto start = std::chrono::steady_clock::now();
    if (hierarchy.levels.empty() ||
        hierarchy.fine_to_coarse.size() + 1 != hierarchy.levels.size()) {
        throw std::invalid_argument("uncoarsen requires a complete hierarchy");
    }
    auto result = refine_hierarchy_device(
        hierarchy, std::forward<P>(coarsest_partition), options);
    for (const auto& level : result.levels) {
        const auto& stats = level.plain;
        const auto& pair_stats = level.pair;
        const auto& cleanup_stats = level.cleanup;
        const auto& timings = level.timings;
        std::cout << std::setprecision(10)
                  << "uncoarsen_level=" << level.level
                  << " vertices=" << level.vertices
                  << " edge_entries=" << level.edge_entries
                  << " projection_cut=" << level.projection_cut
                  << " projection_max_part_weight="
                  << level.projection_max_part_weight
                  << " projection_imbalance=" << level.projection_imbalance
                  << " refine_rounds=" << (stats.rounds + cleanup_stats.rounds)
                  << " refine_proposals="
                  << (stats.proposals + cleanup_stats.proposals)
                  << " refine_accepted="
                  << (stats.accepted + cleanup_stats.accepted)
                  << " refined_cut=" << cleanup_stats.final_cut
                  << " refined_max_part_weight="
                  << cleanup_stats.final_max_part_weight
                  << " pair_candidates=" << pair_stats.candidates
                  << " mutual_pairs=" << pair_stats.mutual_pairs
                  << " pair_accepted=" << pair_stats.accepted
                  << " pair_cut_before=" << pair_stats.cut_before
                  << " pair_cut_after=" << pair_stats.cut_after
                  << " pair_rollback=" << (pair_stats.rollback ? 1 : 0)
                  << " cleanup_cut_after=" << cleanup_stats.final_cut
                  << " graph_h2d_seconds=" << timings.graph_h2d_seconds
                  << " workspace_resize_seconds=" << timings.workspace_resize_seconds
                  << " projection_gpu_seconds="
                  << timings.projection_gpu_seconds
                  << " plain_seconds=" << timings.plain_seconds
                  << " pair_seconds=" << timings.pair_seconds
                  << " cleanup_seconds=" << timings.cleanup_seconds
                  << " verification_seconds="
                  << timings.verification_seconds << '\n';
    }

    const auto total_weight = sum_weights(result.final_part_weights);
    const auto maximum = *std::max_element(
        result.final_part_weights.begin(), result.final_part_weights.end());
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << std::setprecision(10)
              << "final_edge_cut=" << result.final_cut << '\n'
              << "final_cut_ratio="
              << (result.total_edge_weight == 0 ? 0.0 :
                  static_cast<double>(result.final_cut) /
                  result.total_edge_weight) << '\n'
              << "final_max_part_weight=" << maximum << '\n'
              << "final_imbalance="
              << imbalance_of(result.final_part_weights, total_weight) << '\n'
              << "final_d2h_seconds=" << result.final_d2h_seconds << '\n'
              << "uncoarsen_total_seconds=" << seconds << '\n';
    return result;
}

template <typename Types>
std::vector<typename Types::VertexT> uncoarsen(
    const Hierarchy<Types>& hierarchy,
    const std::vector<typename Types::VertexT>& partition, const RefineOptions& options) {
    auto result = uncoarsen_impl<Types>(hierarchy, partition, options);
    return std::move(result.partition);
}

template <typename Types>
DeviceUncoarsenResult<Types> uncoarsen(
    DeviceHierarchy<Types>& hierarchy,
    thrust::device_vector<typename Types::VertexT>&& partition, const RefineOptions& options) {
    return uncoarsen_impl<Types>(hierarchy, std::move(partition), options);
}

template std::vector<ActiveTypes::VertexT> uncoarsen<ActiveTypes>(
    const Hierarchy<ActiveTypes>&,
    const std::vector<ActiveTypes::VertexT>&,
    const RefineOptions&);
template DeviceUncoarsenResult<ActiveTypes> uncoarsen<ActiveTypes>(
    DeviceHierarchy<ActiveTypes>&, thrust::device_vector<ActiveTypes::VertexT>&&,
    const RefineOptions&);

}  // namespace gpart
