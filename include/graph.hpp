#pragma once

#include "graph_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

#include <thrust/device_vector.h>

namespace gpart {

template <typename Types>
struct WeightedGraph {
    using VertexT = typename Types::VertexT;
    using OffsetT = typename Types::OffsetT;
    using WeightT = typename Types::WeightT;

    std::vector<OffsetT> offsets;
    std::vector<VertexT> neighbors;
    std::vector<WeightT> edge_weights;
    std::vector<WeightT> vertex_weights;

    std::int64_t vertices() const {
        return static_cast<std::int64_t>(vertex_weights.size());
    }
    std::int64_t edges() const {
        return static_cast<std::int64_t>(neighbors.size());
    }
};

template <typename Types>
struct DeviceWeightedGraph {
    using VertexT = typename Types::VertexT;
    using OffsetT = typename Types::OffsetT;
    using WeightT = typename Types::WeightT;

    thrust::device_vector<OffsetT> offsets;
    thrust::device_vector<VertexT> neighbors;
    thrust::device_vector<WeightT> edge_weights;
    thrust::device_vector<WeightT> vertex_weights;

    std::int64_t vertices() const {
        return static_cast<std::int64_t>(vertex_weights.size());
    }
    std::int64_t edges() const {
        return static_cast<std::int64_t>(neighbors.size());
    }
};

template <typename Types>
WeightedGraph<Types> load_weighted_graph(
    const std::string& indptr_path, const std::string& indices_path);

template <typename Types>
void validate_weighted_shape(
    const WeightedGraph<Types>& graph, bool allow_self_loops);

template <typename Types>
void validate_weighted_csr(
    const WeightedGraph<Types>& graph, bool allow_self_loops);

template <typename Types>
std::uint64_t host_cut(
    const WeightedGraph<Types>& graph,
    const std::vector<typename Types::VertexT>& labels);

template <typename Types>
DeviceWeightedGraph<Types> make_device_weighted(
    const WeightedGraph<Types>& graph);

template <typename Types>
WeightedGraph<Types> copy_device_weighted(
    const DeviceWeightedGraph<Types>& graph);

}  // namespace gpart
