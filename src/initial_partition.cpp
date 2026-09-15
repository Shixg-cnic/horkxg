#include "initial_partition.hpp"

#include "graph_types.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <metis.h>

namespace gpart {
static_assert(
    sizeof(idx_t) * 8 == IDXTYPEWIDTH,
    "METIS idx_t does not match the configured IDXTYPEWIDTH");

namespace {

template <typename ValueT>
idx_t checked_idx(ValueT value, const char* field) {
    static_assert(std::is_integral_v<ValueT>);
    bool fits = true;
    if constexpr (std::is_signed_v<ValueT>) {
        fits = value >= 0;
        if constexpr (sizeof(ValueT) > sizeof(idx_t)) {
            fits = fits && value <= static_cast<ValueT>(
                std::numeric_limits<idx_t>::max());
        } else {
            fits = fits && static_cast<std::uint64_t>(value) <=
                static_cast<std::uint64_t>(std::numeric_limits<idx_t>::max());
        }
    } else {
        fits = value <= static_cast<std::make_unsigned_t<idx_t>>(
            std::numeric_limits<idx_t>::max());
    }
    if (!fits) {
        throw std::overflow_error(
            std::string(field) + " cannot be represented by METIS idx_t");
    }
    return static_cast<idx_t>(value);
}

void add_checked_total(
    std::uint64_t& total, std::uint64_t value, const char* field) {
    const auto metis_max = static_cast<std::uint64_t>(
        std::numeric_limits<idx_t>::max());
    if (value > metis_max || total > metis_max - value) {
        throw std::overflow_error(
            std::string(field) + " exceeds METIS idx_t accumulation range");
    }
    total += value;
}

}  // namespace

template <typename Types>
std::vector<typename Types::VertexT> initial_partition(
    const WeightedGraph<Types>& graph,
    int parts,
    double imbalance_ratio) {
    using VertexT = typename Types::VertexT;
    const auto start = std::chrono::steady_clock::now();
    const auto vertex_count = graph.vertices();
    const auto edge_count = graph.edges();
    if (vertex_count <= 0) {
        throw std::invalid_argument("initial partition requires a nonempty graph");
    }
    if (parts < 2 || parts > vertex_count) {
        throw std::invalid_argument("invalid initial partition part count");
    }
    if (!std::isfinite(imbalance_ratio) || imbalance_ratio < 1.0 ||
        imbalance_ratio > static_cast<double>(
                              std::numeric_limits<real_t>::max())) {
        throw std::invalid_argument("invalid initial partition imbalance ratio");
    }
    if (graph.offsets.size() != graph.vertex_weights.size() + 1 ||
        graph.offsets.empty() || graph.offsets.front() != 0 ||
        graph.neighbors.size() != graph.edge_weights.size()) {
        throw std::invalid_argument("invalid weighted CSR shape for METIS");
    }

    idx_t nvtxs = checked_idx(vertex_count, "vertex count");
    idx_t ncon = 1;
    idx_t nparts = checked_idx(parts, "part count");
    std::vector<idx_t> xadj(graph.offsets.size());
    std::vector<idx_t> adjncy(graph.neighbors.size());
    std::vector<idx_t> vertex_weights(graph.vertex_weights.size());
    std::vector<idx_t> edge_weights(graph.edge_weights.size());

    for (std::size_t i = 0; i < graph.offsets.size(); ++i) {
        const auto offset = graph.offsets[i];
        if (i > 0 && offset < graph.offsets[i - 1]) {
            throw std::invalid_argument("weighted CSR offsets are not monotone");
        }
        xadj[i] = checked_idx(offset, "CSR offset");
    }
    if (xadj.back() != checked_idx(edge_count, "edge-entry count")) {
        throw std::invalid_argument("weighted CSR offsets do not match edges");
    }

    for (std::size_t i = 0; i < graph.neighbors.size(); ++i) {
        const auto neighbor = graph.neighbors[i];
        if (neighbor < 0 || static_cast<std::int64_t>(neighbor) >= vertex_count) {
            throw std::invalid_argument("weighted CSR contains an invalid neighbor");
        }
        adjncy[i] = checked_idx(neighbor, "neighbor id");
    }

    std::uint64_t total_vertex_weight = 0;
    for (std::size_t i = 0; i < graph.vertex_weights.size(); ++i) {
        const auto weight = static_cast<std::uint64_t>(graph.vertex_weights[i]);
        vertex_weights[i] = checked_idx(weight, "vertex weight");
        add_checked_total(total_vertex_weight, weight, "total vertex weight");
    }
    std::uint64_t total_edge_weight = 0;
    for (std::size_t i = 0; i < graph.edge_weights.size(); ++i) {
        const auto weight = static_cast<std::uint64_t>(graph.edge_weights[i]);
        edge_weights[i] = checked_idx(weight, "edge weight");
        add_checked_total(total_edge_weight, weight, "total edge weight");
    }

    real_t ubvec = static_cast<real_t>(imbalance_ratio);
    if (!std::isfinite(static_cast<double>(ubvec)) || ubvec < real_t{1}) {
        throw std::overflow_error(
            "imbalance ratio cannot be represented by METIS real_t");
    }
    idx_t edgecut = 0;
    std::vector<idx_t> metis_partition(
        static_cast<std::size_t>(vertex_count), idx_t{-1});
    const int status = METIS_PartGraphKway(
        &nvtxs, &ncon, xadj.data(), adjncy.data(), vertex_weights.data(),
        nullptr, edge_weights.data(), &nparts, nullptr, &ubvec, nullptr,
        &edgecut, metis_partition.data());
    if (status != METIS_OK) {
        throw std::runtime_error(
            "METIS_PartGraphKway failed with status " +
            std::to_string(status));
    }

    std::vector<VertexT> partition(static_cast<std::size_t>(vertex_count));
    for (std::size_t v = 0; v < partition.size(); ++v) {
        const auto part = metis_partition[v];
        if (part < 0 || part >= nparts) {
            throw std::runtime_error("METIS returned an invalid part id");
        }
        partition[v] = static_cast<VertexT>(part);
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "coarsest_vertices=" << vertex_count << '\n'
              << "coarsest_edge_entries=" << edge_count << '\n'
              << "initial_partition_parts=" << parts << '\n'
              << "initial_partition_edgecut=" << edgecut << '\n'
              << "initial_partition_seconds=" << seconds << '\n';
    return partition;
}

template std::vector<StandardTypes::VertexT> initial_partition<StandardTypes>(
    const WeightedGraph<StandardTypes>&, int, double);
template std::vector<BigTypes::VertexT> initial_partition<BigTypes>(
    const WeightedGraph<BigTypes>&, int, double);

}  // namespace gpart
