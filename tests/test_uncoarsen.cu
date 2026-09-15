#include "graph.hpp"
#include "graph_types.hpp"
#include "hierarchy.hpp"
#include "refine.hpp"
#include "uncoarsen.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

template <typename Types>
gpart::Hierarchy<Types> make_hierarchy() {
    using OffsetT = typename Types::OffsetT;
    using VertexT = typename Types::VertexT;
    using WeightT = typename Types::WeightT;
    gpart::WeightedGraph<Types> fine;
    fine.offsets = {OffsetT{0}, OffsetT{1}, OffsetT{3}, OffsetT{5}, OffsetT{6}};
    fine.neighbors = {
        VertexT{1}, VertexT{0}, VertexT{2},
        VertexT{1}, VertexT{3}, VertexT{2}};
    fine.edge_weights = {
        WeightT{1}, WeightT{1}, WeightT{5},
        WeightT{5}, WeightT{10}, WeightT{10}};
    fine.vertex_weights = {WeightT{1}, WeightT{1}, WeightT{1}, WeightT{1}};

    gpart::WeightedGraph<Types> coarse;
    coarse.offsets = {OffsetT{0}, OffsetT{1}, OffsetT{2}};
    coarse.neighbors = {VertexT{1}, VertexT{0}};
    coarse.edge_weights = {WeightT{5}, WeightT{5}};
    coarse.vertex_weights = {WeightT{2}, WeightT{2}};

    gpart::Hierarchy<Types> hierarchy;
    hierarchy.levels.push_back(std::move(fine));
    hierarchy.levels.push_back(std::move(coarse));
    hierarchy.fine_to_coarse.push_back(
        {VertexT{0}, VertexT{0}, VertexT{1}, VertexT{1}});
    return hierarchy;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
    try {
        using Types = gpart::ActiveTypes;
        using VertexT = typename Types::VertexT;
        auto hierarchy = make_hierarchy<Types>();
        const std::vector<VertexT> coarse_partition{VertexT{0}, VertexT{1}};
        const auto& map = hierarchy.fine_to_coarse.front();
        std::vector<VertexT> projected(map.size());
        for (std::size_t v = 0; v < map.size(); ++v) {
            projected[v] = coarse_partition[map[v]];
        }
        require(
            projected == std::vector<VertexT>({0, 0, 1, 1}),
            "projection labels are wrong");
        require(
            gpart::host_cut(hierarchy.levels.back(), coarse_partition) == 5 &&
            gpart::host_cut(hierarchy.levels.front(), projected) == 5,
            "projection did not preserve cut");
        require(
            hierarchy.levels.back().vertex_weights[0] == 2 &&
            hierarchy.levels.back().vertex_weights[1] == 2,
            "coarse part weights are wrong");
        std::uint64_t projected_weight[2] = {0, 0};
        for (std::size_t v = 0; v < projected.size(); ++v) {
            projected_weight[projected[v]] +=
                hierarchy.levels.front().vertex_weights[v];
        }
        require(
            projected_weight[0] == 2 && projected_weight[1] == 2,
            "projection did not preserve part weights");

        gpart::RefineOptions options;
        options.parts = 2;
        options.imbalance_ratio = 1.10;
        options.seed = 0;
        options.max_rounds = 4;
        options.strict_verify = true;
        const auto final_partition = gpart::uncoarsen(
            hierarchy, coarse_partition, options);
        require(final_partition.size() == 4, "final partition length is wrong");
        for (const auto part : final_partition) {
            require(part >= 0 && part < 2, "final part id is invalid");
        }
        const auto final_cut = gpart::host_cut(
            hierarchy.levels.front(), final_partition);
        require(final_cut <= 5, "refinement increased cut");
        std::uint64_t final_weight[2] = {0, 0};
        for (std::size_t v = 0; v < final_partition.size(); ++v) {
            final_weight[final_partition[v]] +=
                hierarchy.levels.front().vertex_weights[v];
        }
        require(
            std::max(final_weight[0], final_weight[1]) <= 3,
            "refinement violated balance");
        std::cout << "uncoarsen_small_ok=1 graph_type="
                  << gpart::kActiveGraphName << " final_cut=" << final_cut
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_uncoarsen: " << error.what() << '\n';
        return 1;
    }
}
