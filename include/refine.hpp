#pragma once

#include "graph.hpp"

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

}  // namespace gpart
