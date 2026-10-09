#include "initial_partition.hpp"
#include "partition_capacity.hpp"
#include <thrust/copy.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using Types = gpart::ActiveTypes;
using W = Types::WeightT;
void check(const gpart::WeightedGraph<Types>& graph, int parts, bool zero_cut) {
    auto device = gpart::make_device_weighted(graph);
    auto a = gpart::initial_partition_gpu(device, parts, 1.03, 0);
    auto b = gpart::initial_partition_gpu(device, parts, 1.03, 0);
    std::vector<int> labels(a.size()), again(b.size());
    thrust::copy(a.begin(), a.end(), labels.begin());
    thrust::copy(b.begin(), b.end(), again.begin());
    if (labels != again) throw std::runtime_error("GPU initialization is not repeatable");
    std::vector<std::uint64_t> weights(parts, 0);
    std::uint64_t total = 0;
    for (std::size_t v = 0; v < labels.size(); ++v) {
        if (labels[v] < 0 || labels[v] >= parts) throw std::runtime_error("invalid label");
        weights[labels[v]] += graph.vertex_weights[v];
        total += graph.vertex_weights[v];
    }
    auto capacity = gpart::partition_capacity(total, parts, 1.03);
    if (*std::max_element(weights.begin(), weights.end()) > capacity)
        throw std::runtime_error("capacity violation");
    if (zero_cut && gpart::host_cut(graph, labels) != 0)
        throw std::runtime_error("disconnected community quality regression");
}
int main() {
    gpart::WeightedGraph<Types> graph;
    graph.offsets.push_back(0);
    // Two weighted cliques: the spectral solution must preserve components.
    for (int v = 0; v < 8; ++v) {
        for (int u = 0; u < 8; ++u) if (u != v && u / 4 == v / 4) {
            graph.neighbors.push_back(u); graph.edge_weights.push_back(W{3});
        }
        graph.offsets.push_back(graph.neighbors.size());
        graph.vertex_weights.push_back(W{2});
    }
    check(graph, 2, true);
    if constexpr (sizeof(W) == 8) {
        graph.vertex_weights.assign(8, static_cast<W>(std::uint64_t{1} << 40));
        graph.edge_weights.assign(graph.neighbors.size(), static_cast<W>(std::uint64_t{1} << 40));
        check(graph, 2, true);
    }
    // Isolated vertices and a non-power-of-two partition count.
    graph.offsets.assign(13, 0); graph.neighbors.clear(); graph.edge_weights.clear();
    graph.vertex_weights.assign(12, W{1});
    check(graph, 3, true);
    // Partial blocks, hubs, duplicate edges and an isolated tail vertex.
    // Integer loads and repeatability exercise the fused moment reductions.
    graph.offsets.assign(1, 0);
    graph.vertex_weights.assign(259, W{1});
    for (int v = 0; v < 259; ++v) {
        std::vector<int> row;
        if (v == 0) for (int u = 1; u < 258; ++u) { row.push_back(u); row.push_back(u); }
        if (v > 0 && v < 258) { row.push_back(0); row.push_back(0); }
        if (v > 1 && v < 258) row.push_back(v - 1);
        if (v > 0 && v < 257) row.push_back(v + 1);
        std::sort(row.begin(), row.end());
        graph.neighbors.insert(graph.neighbors.end(), row.begin(), row.end());
        graph.edge_weights.insert(graph.edge_weights.end(), row.size(), W{7});
        graph.offsets.push_back(graph.neighbors.size());
    }
    for (int k : {2, 3, 4, 8}) check(graph, k, false);
    graph.offsets.assign(257, 0); graph.neighbors.clear(); graph.edge_weights.clear();
    graph.vertex_weights.assign(256, W{1});
    check(graph, 32, true);
    // No initializer can place this indivisible vertex under capacity.
    graph.vertex_weights[0] = W{1000};
    bool rejected = false;
    try { auto d = gpart::make_device_weighted(graph); gpart::initial_partition_gpu(d, 3, 1.10); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("overweight vertex was not rejected");
    graph.offsets.assign(5, 0); graph.vertex_weights.assign(4, W{1});
    rejected = false;
    try { auto d = gpart::make_device_weighted(graph); gpart::initial_partition_gpu(d, 3, 1.03); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("impossible integer capacity was not rejected");
    // No split of three indivisible weight-two vertices can fit capacity three.
    graph.offsets.assign(4, 0); graph.vertex_weights.assign(3, W{2});
    rejected = false;
    try { auto d = gpart::make_device_weighted(graph); gpart::initial_partition_gpu(d, 2, 1.03); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("infeasible weighted split was not rejected");
    if constexpr (sizeof(W) == 8) {
        graph.offsets = {0, 1, 2}; graph.neighbors = {1, 0};
        graph.vertex_weights.assign(2, W{1});
        graph.edge_weights.assign(2, std::numeric_limits<W>::max());
        rejected = false;
        try { auto d = gpart::make_device_weighted(graph); gpart::initial_partition_gpu(d, 2, 1.03); }
        catch (const std::overflow_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("edge weight overflow was not rejected");
    }
    std::cout << "GPU initial correctness passed: " << gpart::kActiveGraphName << '\n';
}
