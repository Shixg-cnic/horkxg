#pragma once

#include "graph.hpp"

#include <vector>

namespace gpart {

template <typename Types>
std::vector<typename Types::VertexT> initial_partition(
    const WeightedGraph<Types>& graph,
    int parts,
    double imbalance_ratio);

}  // namespace gpart
