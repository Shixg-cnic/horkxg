#pragma once

#include "graph.hpp"

#include <vector>

namespace gpart {

template <typename Types>
std::vector<typename Types::VertexT> initial_partition(
    const WeightedGraph<Types>& graph,
    int parts,
    double imbalance_ratio);

template <typename Types>
thrust::device_vector<typename Types::VertexT> initial_partition(
    const DeviceWeightedGraph<Types>& graph, int parts, double imbalance_ratio) {
    auto host = copy_device_weighted(graph);
    auto labels = initial_partition(host, parts, imbalance_ratio);
    return thrust::device_vector<typename Types::VertexT>(labels);
}

}  // namespace gpart
