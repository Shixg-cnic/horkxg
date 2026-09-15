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

template <typename Types>
void refine_partition(
    const WeightedGraph<Types>& graph,
    std::vector<typename Types::VertexT>& partition,
    const RefineOptions& options,
    RefineStats* stats = nullptr);

}  // namespace gpart
