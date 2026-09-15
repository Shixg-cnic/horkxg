#include "graph.hpp"
#include "graph_types.hpp"
#include "initial_partition.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <typename Types>
void run_case() {
    using OffsetT = typename Types::OffsetT;
    using VertexT = typename Types::VertexT;
    using WeightT = typename Types::WeightT;
    gpart::WeightedGraph<Types> graph;
    graph.offsets = {
        OffsetT{0}, OffsetT{3}, OffsetT{6}, OffsetT{9}, OffsetT{12},
        OffsetT{15}, OffsetT{18}, OffsetT{21}, OffsetT{24}};
    graph.neighbors = {
        VertexT{1}, VertexT{2}, VertexT{3},
        VertexT{0}, VertexT{2}, VertexT{3},
        VertexT{0}, VertexT{1}, VertexT{3},
        VertexT{0}, VertexT{1}, VertexT{2},
        VertexT{5}, VertexT{6}, VertexT{7},
        VertexT{4}, VertexT{6}, VertexT{7},
        VertexT{4}, VertexT{5}, VertexT{7},
        VertexT{4}, VertexT{5}, VertexT{6}};
    graph.edge_weights.assign(graph.neighbors.size(), WeightT{3});
    graph.vertex_weights.assign(8, WeightT{1});

    constexpr int parts = 2;
    constexpr double imbalance = 1.10;
    const auto partition =
        gpart::initial_partition<Types>(graph, parts, imbalance);
    if (partition.size() != graph.vertex_weights.size()) {
        throw std::runtime_error("initial partition has the wrong length");
    }
    std::vector<std::uint64_t> part_weights(parts, 0);
    for (std::size_t v = 0; v < partition.size(); ++v) {
        const auto part = partition[v];
        if (part < 0 || part >= parts) {
            throw std::runtime_error("initial partition has an invalid part id");
        }
        part_weights[static_cast<std::size_t>(part)] +=
            static_cast<std::uint64_t>(graph.vertex_weights[v]);
    }
    std::uint64_t total = 0;
    for (const auto weight : graph.vertex_weights) {
        total += static_cast<std::uint64_t>(weight);
    }
    const auto ideal_ceiling = (total + parts - 1) / parts;
    const auto capacity = static_cast<std::uint64_t>(
        std::floor(imbalance * static_cast<double>(ideal_ceiling)));
    if (*std::max_element(part_weights.begin(), part_weights.end()) > capacity) {
        throw std::runtime_error("initial partition violates imbalance bound");
    }
}

}  // namespace

int main() {
    run_case<gpart::StandardTypes>();
    run_case<gpart::BigTypes>();
    std::cout << "initial partition correctness test passed\n";
    return 0;
}
