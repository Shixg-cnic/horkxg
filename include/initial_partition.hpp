#pragma once

#include "graph.hpp"

namespace gpart {

// GPU spectral recursive bisection used by the production partitioner.
// Requires a validated symmetric nonnegative weighted CSR, just like coarsen.
template <typename Types>
thrust::device_vector<typename Types::VertexT> initial_partition_gpu(
    const DeviceWeightedGraph<Types>& graph, int parts, double imbalance_ratio,
    std::uint32_t seed = 0);

}  // namespace gpart
