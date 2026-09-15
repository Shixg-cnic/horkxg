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

template <typename Types>
std::vector<std::uint64_t> partition_weights(
    const WeightedGraph<Types>& graph,
    const std::vector<typename Types::VertexT>& partition,
    int parts) {
    if (partition.size() != static_cast<std::size_t>(graph.vertices())) {
        throw std::invalid_argument("partition length does not match graph");
    }
    std::vector<std::uint64_t> weights(static_cast<std::size_t>(parts), 0);
    for (std::size_t vertex = 0; vertex < partition.size(); ++vertex) {
        const auto part = partition[vertex];
        if (part < 0 || part >= parts) {
            throw std::invalid_argument("partition contains an invalid part id");
        }
        const auto weight = static_cast<std::uint64_t>(
            graph.vertex_weights[vertex]);
        if (weights[part] > std::numeric_limits<std::uint64_t>::max() - weight) {
            throw std::overflow_error("partition weight overflow");
        }
        weights[part] += weight;
    }
    return weights;
}

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

template <typename Types>
std::uint64_t undirected_edge_weight(const WeightedGraph<Types>& graph) {
    std::uint64_t directed = 0;
    for (const auto value : graph.edge_weights) {
        const auto weight = static_cast<std::uint64_t>(value);
        if (directed > std::numeric_limits<std::uint64_t>::max() - weight) {
            throw std::overflow_error("edge weight sum overflow");
        }
        directed += weight;
    }
    if ((directed & 1U) != 0) {
        throw std::runtime_error("symmetric graph has odd directed edge weight");
    }
    return directed / 2;
}

}  // namespace

template <typename Types>
std::vector<typename Types::VertexT> uncoarsen(
    const Hierarchy<Types>& hierarchy,
    const std::vector<typename Types::VertexT>& coarsest_partition,
    const RefineOptions& options) {
    using VertexT = typename Types::VertexT;
    const auto start = std::chrono::steady_clock::now();
    if (hierarchy.levels.empty() ||
        hierarchy.fine_to_coarse.size() + 1 != hierarchy.levels.size()) {
        throw std::invalid_argument("uncoarsen requires a complete hierarchy");
    }
    auto partition = coarsest_partition;
    partition_weights(hierarchy.levels.back(), partition, options.parts);

    for (std::size_t coarse_level = hierarchy.levels.size() - 1;
         coarse_level > 0; --coarse_level) {
        const auto fine_level = coarse_level - 1;
        const auto& coarse = hierarchy.levels[coarse_level];
        const auto& fine = hierarchy.levels[fine_level];
        const auto& map = hierarchy.fine_to_coarse[fine_level];
        if (map.size() != static_cast<std::size_t>(fine.vertices())) {
            throw std::invalid_argument("fine-to-coarse map has wrong length");
        }
        const auto coarse_weights = partition_weights(
            coarse, partition, options.parts);
        const auto coarse_cut = host_cut(coarse, partition);

        std::vector<VertexT> projected(map.size());
        for (std::size_t vertex = 0; vertex < map.size(); ++vertex) {
            const auto coarse_vertex = map[vertex];
            if (coarse_vertex < 0 || coarse_vertex >= coarse.vertices()) {
                throw std::invalid_argument(
                    "fine-to-coarse map contains an invalid id");
            }
            projected[vertex] = partition[coarse_vertex];
        }
        const auto projected_weights = partition_weights(
            fine, projected, options.parts);
        const auto projected_cut = host_cut(fine, projected);
        if (options.strict_verify &&
            (projected_cut != coarse_cut || projected_weights != coarse_weights)) {
            throw std::runtime_error(
                "projection failed cut or part-weight preservation");
        }

        const auto projected_total = sum_weights(projected_weights);
        const auto projected_max = *std::max_element(
            projected_weights.begin(), projected_weights.end());
        RefineStats stats;
        refine_partition(fine, projected, options, &stats);
        PairRefineStats pair_stats;
        coordinated_pair_escape(
            fine, projected, options, static_cast<int>(fine_level),
            &pair_stats);
        auto cleanup_options = options;
        cleanup_options.max_rounds = 1;
        RefineStats cleanup_stats;
        refine_partition(
            fine, projected, cleanup_options, &cleanup_stats);
        std::cout << std::setprecision(10)
                  << "uncoarsen_level=" << fine_level
                  << " vertices=" << fine.vertices()
                  << " edge_entries=" << fine.edges()
                  << " projection_cut=" << projected_cut
                  << " projection_max_part_weight=" << projected_max
                  << " projection_imbalance="
                  << imbalance_of(projected_weights, projected_total)
                  << " refine_rounds=" << (stats.rounds + cleanup_stats.rounds)
                  << " refine_proposals="
                  << (stats.proposals + cleanup_stats.proposals)
                  << " refine_accepted="
                  << (stats.accepted + cleanup_stats.accepted)
                  << " refined_cut=" << cleanup_stats.final_cut
                  << " refined_max_part_weight="
                  << cleanup_stats.final_max_part_weight
                  << " refine_seconds="
                  << (stats.seconds + cleanup_stats.seconds)
                  << " pair_candidates=" << pair_stats.candidates
                  << " mutual_pairs=" << pair_stats.mutual_pairs
                  << " pair_accepted=" << pair_stats.accepted
                  << " pair_cut_before=" << pair_stats.cut_before
                  << " pair_cut_after=" << pair_stats.cut_after
                  << " pair_rollback=" << (pair_stats.rollback ? 1 : 0)
                  << " cleanup_cut_after=" << cleanup_stats.final_cut << '\n';
        partition = std::move(projected);
    }

    const auto& original = hierarchy.levels.front();
    const auto final_weights = partition_weights(
        original, partition, options.parts);
    const auto total_weight = sum_weights(final_weights);
    const auto final_cut = host_cut(original, partition);
    const auto total_edge_weight = undirected_edge_weight(original);
    const auto maximum = *std::max_element(
        final_weights.begin(), final_weights.end());
    const auto seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << std::setprecision(10)
              << "final_edge_cut=" << final_cut << '\n'
              << "final_cut_ratio="
              << (total_edge_weight == 0 ? 0.0 :
                  static_cast<double>(final_cut) / total_edge_weight) << '\n'
              << "final_max_part_weight=" << maximum << '\n'
              << "final_imbalance="
              << imbalance_of(final_weights, total_weight) << '\n'
              << "uncoarsen_total_seconds=" << seconds << '\n';
    return partition;
}

template std::vector<ActiveTypes::VertexT> uncoarsen<ActiveTypes>(
    const Hierarchy<ActiveTypes>&,
    const std::vector<ActiveTypes::VertexT>&,
    const RefineOptions&);

}  // namespace gpart
