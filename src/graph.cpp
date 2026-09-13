#include "graph.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include <thrust/copy.h>

namespace gpart {
namespace {

std::int64_t element_count(const std::string& path, std::size_t element_size) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path);
    const auto bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(element_size) != 0) {
        throw std::runtime_error("invalid binary size " + path);
    }
    return static_cast<std::int64_t>(
        bytes / static_cast<std::streamoff>(element_size));
}

template <typename T>
std::vector<T> read_all(const std::string& path, std::int64_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    std::vector<T> values(static_cast<std::size_t>(count));
    if (count > 0) {
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(count * sizeof(T)));
    }
    if (!input) throw std::runtime_error("cannot read " + path);
    return values;
}

[[noreturn]] void standard_limit_error() {
    throw std::runtime_error(
        "graph exceeds standard CSR limits; use gpart_coarsen_big");
}

}  // namespace

template <typename Types>
WeightedGraph<Types> load_weighted_graph(
    const std::string& indptr_path, const std::string& indices_path) {
    using VertexT = typename Types::VertexT;
    using OffsetT = typename Types::OffsetT;
    using WeightT = typename Types::WeightT;

    const auto offset_count = element_count(indptr_path, sizeof(std::int64_t));
    const auto edge_count = element_count(indices_path, sizeof(std::int64_t));
    if (offset_count < 1) throw std::runtime_error("empty CSR indptr");
    const auto vertex_count = offset_count - 1;

    if constexpr (std::is_same_v<Types, StandardTypes>) {
        if (vertex_count > std::numeric_limits<VertexT>::max() ||
            edge_count > std::numeric_limits<OffsetT>::max() ||
            static_cast<std::uint64_t>(vertex_count) >
                std::numeric_limits<WeightT>::max() ||
            static_cast<std::uint64_t>(edge_count) >
                std::numeric_limits<WeightT>::max()) {
            standard_limit_error();
        }
    } else if (vertex_count > std::numeric_limits<VertexT>::max()) {
        throw std::runtime_error("graph vertex ids exceed configured graph type");
    }

    const auto disk_offsets = read_all<std::int64_t>(indptr_path, offset_count);
    if (disk_offsets.front() != 0 || disk_offsets.back() != edge_count) {
        throw std::runtime_error("CSR offsets do not match indices");
    }

    WeightedGraph<Types> graph;
    graph.offsets.resize(static_cast<std::size_t>(offset_count));
    for (std::int64_t i = 0; i < offset_count; ++i) {
        const auto offset = disk_offsets[static_cast<std::size_t>(i)];
        if (offset < 0 || (i > 0 && offset < disk_offsets[static_cast<std::size_t>(i - 1)])) {
            throw std::runtime_error("CSR offsets are not monotone");
        }
        if (offset > static_cast<std::int64_t>(
                         std::numeric_limits<OffsetT>::max())) {
            if constexpr (std::is_same_v<Types, StandardTypes>) {
                standard_limit_error();
            }
            throw std::runtime_error("CSR offset exceeds configured graph type");
        }
        graph.offsets[static_cast<std::size_t>(i)] =
            static_cast<OffsetT>(offset);
    }

    std::ifstream input(indices_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + indices_path);
    graph.neighbors.resize(static_cast<std::size_t>(edge_count));
    constexpr std::size_t chunk = 1U << 20;
    std::vector<std::int64_t> raw(chunk);
    std::int64_t done = 0;
    while (done < edge_count) {
        const auto take = static_cast<std::size_t>(
            std::min<std::int64_t>(chunk, edge_count - done));
        input.read(reinterpret_cast<char*>(raw.data()),
                   static_cast<std::streamsize>(take * sizeof(std::int64_t)));
        if (!input) throw std::runtime_error("cannot read " + indices_path);
        for (std::size_t i = 0; i < take; ++i) {
            const auto neighbor = raw[i];
            if (neighbor < 0 || neighbor >= vertex_count) {
                throw std::runtime_error("CSR contains an invalid neighbor");
            }
            graph.neighbors[static_cast<std::size_t>(done) + i] =
                static_cast<VertexT>(neighbor);
        }
        done += static_cast<std::int64_t>(take);
    }
    graph.edge_weights.assign(static_cast<std::size_t>(edge_count), WeightT{1});
    graph.vertex_weights.assign(static_cast<std::size_t>(vertex_count), WeightT{1});
    std::cout << "vertices=" << vertex_count << " edge_entries=" << edge_count
              << " device_count=1\n";
    return graph;
}

template <typename Types>
void validate_weighted_shape(
    const WeightedGraph<Types>& graph, bool allow_self_loops) {
    using OffsetT = typename Types::OffsetT;
    if (graph.offsets.size() != graph.vertex_weights.size() + 1 ||
        graph.offsets.empty() || graph.offsets.front() != OffsetT{0} ||
        static_cast<std::int64_t>(graph.offsets.back()) != graph.edges() ||
        graph.neighbors.size() != graph.edge_weights.size()) {
        throw std::runtime_error("invalid weighted CSR shape");
    }
    for (std::int64_t v = 0; v < graph.vertices(); ++v) {
        const auto begin = graph.offsets[static_cast<std::size_t>(v)];
        const auto end = graph.offsets[static_cast<std::size_t>(v + 1)];
        if (begin > end) {
            throw std::runtime_error("weighted CSR offsets are not monotone");
        }
        for (auto e = begin; e < end; ++e) {
            const auto u = graph.neighbors[static_cast<std::size_t>(e)];
            if (u < 0 || u >= graph.vertices()) {
                throw std::runtime_error("weighted CSR contains an invalid neighbor");
            }
            if (!allow_self_loops && u == v) {
                throw std::runtime_error("contracted graph contains a self loop");
            }
        }
    }
}

template <typename Types>
void validate_weighted_csr(
    const WeightedGraph<Types>& graph, bool allow_self_loops) {
    validate_weighted_shape(graph, allow_self_loops);
    for (std::int64_t v = 0; v < graph.vertices(); ++v) {
        const auto begin = graph.offsets[static_cast<std::size_t>(v)];
        const auto end = graph.offsets[static_cast<std::size_t>(v + 1)];
        for (auto e = begin; e < end; ++e) {
            const auto u = graph.neighbors[static_cast<std::size_t>(e)];
            const auto reverse_begin = graph.offsets[static_cast<std::size_t>(u)];
            const auto reverse_end = graph.offsets[static_cast<std::size_t>(u + 1)];
            const auto reverse = std::lower_bound(
                graph.neighbors.begin() + reverse_begin,
                graph.neighbors.begin() + reverse_end,
                static_cast<typename Types::VertexT>(v));
            if (reverse == graph.neighbors.begin() + reverse_end || *reverse != v) {
                throw std::runtime_error("weighted CSR is not bidirectional");
            }
            const auto reverse_edge = static_cast<std::size_t>(
                reverse - graph.neighbors.begin());
            if (graph.edge_weights[static_cast<std::size_t>(e)] !=
                graph.edge_weights[reverse_edge]) {
                throw std::runtime_error("weighted CSR reverse edge weights differ");
            }
        }
    }
}

template <typename Types>
std::uint64_t host_cut(
    const WeightedGraph<Types>& graph,
    const std::vector<typename Types::VertexT>& labels) {
    std::uint64_t directed = 0;
    for (std::int64_t v = 0; v < graph.vertices(); ++v) {
        for (auto e = graph.offsets[static_cast<std::size_t>(v)];
             e < graph.offsets[static_cast<std::size_t>(v + 1)]; ++e) {
            if (labels[static_cast<std::size_t>(v)] !=
                labels[static_cast<std::size_t>(
                    graph.neighbors[static_cast<std::size_t>(e)])]) {
                directed += static_cast<std::uint64_t>(
                    graph.edge_weights[static_cast<std::size_t>(e)]);
            }
        }
    }
    if (directed & 1ULL) {
        throw std::runtime_error("weighted CSR cut is not symmetric");
    }
    return directed / 2;
}

template <typename Types>
DeviceWeightedGraph<Types> make_device_weighted(
    const WeightedGraph<Types>& graph) {
    DeviceWeightedGraph<Types> out;
    out.offsets = graph.offsets;
    out.neighbors = graph.neighbors;
    out.edge_weights = graph.edge_weights;
    out.vertex_weights = graph.vertex_weights;
    return out;
}

template <typename Types>
WeightedGraph<Types> copy_device_weighted(
    const DeviceWeightedGraph<Types>& graph) {
    WeightedGraph<Types> out;
    out.offsets.resize(graph.offsets.size());
    out.neighbors.resize(graph.neighbors.size());
    out.edge_weights.resize(graph.edge_weights.size());
    out.vertex_weights.resize(graph.vertex_weights.size());
    thrust::copy(graph.offsets.begin(), graph.offsets.end(), out.offsets.begin());
    thrust::copy(graph.neighbors.begin(), graph.neighbors.end(), out.neighbors.begin());
    thrust::copy(
        graph.edge_weights.begin(), graph.edge_weights.end(), out.edge_weights.begin());
    thrust::copy(
        graph.vertex_weights.begin(), graph.vertex_weights.end(),
        out.vertex_weights.begin());
    return out;
}

template WeightedGraph<ActiveTypes> load_weighted_graph<ActiveTypes>(
    const std::string&, const std::string&);
template void validate_weighted_shape<ActiveTypes>(
    const WeightedGraph<ActiveTypes>&, bool);
template void validate_weighted_csr<ActiveTypes>(
    const WeightedGraph<ActiveTypes>&, bool);
template std::uint64_t host_cut<ActiveTypes>(
    const WeightedGraph<ActiveTypes>&,
    const std::vector<typename ActiveTypes::VertexT>&);
template DeviceWeightedGraph<ActiveTypes> make_device_weighted<ActiveTypes>(
    const WeightedGraph<ActiveTypes>&);
template WeightedGraph<ActiveTypes> copy_device_weighted<ActiveTypes>(
    const DeviceWeightedGraph<ActiveTypes>&);

}  // namespace gpart
