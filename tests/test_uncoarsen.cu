#include "graph.hpp"
#include "graph_types.hpp"
#include "hierarchy.hpp"
#include "refine.hpp"
#include "uncoarsen.hpp"
#include "coarsen.hpp"
#include "detail/scratch_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
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

template <typename Types>
gpart::WeightedGraph<Types> make_pair_escape_graph() {
    using OffsetT = typename Types::OffsetT;
    using VertexT = typename Types::VertexT;
    using WeightT = typename Types::WeightT;
    gpart::WeightedGraph<Types> graph;
    graph.offsets = {
        OffsetT{0}, OffsetT{2}, OffsetT{4}, OffsetT{6},
        OffsetT{8}, OffsetT{9}, OffsetT{10}};
    graph.neighbors = {
        VertexT{1}, VertexT{2}, VertexT{0}, VertexT{3},
        VertexT{0}, VertexT{3}, VertexT{1}, VertexT{2},
        VertexT{5}, VertexT{4}};
    graph.edge_weights = {
        WeightT{3}, WeightT{2}, WeightT{3}, WeightT{2},
        WeightT{2}, WeightT{3}, WeightT{2}, WeightT{3},
        WeightT{1}, WeightT{1}};
    graph.vertex_weights.assign(6, WeightT{1});
    return graph;
}

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_scratch_pool_lifetime() {
    const auto previous = gpart::detail::active_scratch_pool;
    gpart::detail::ScratchVector<int> retained(5, 2);
    {
        gpart::detail::ScratchPoolSession nested;
        gpart::detail::ScratchVector<int> temporary(17, 9);
        retained = std::move(temporary);
        nested.finish();
    }
    require(gpart::detail::active_scratch_pool == previous,
            "scratch session did not restore its caller's pool");
    retained.resize(33, 9);
    std::vector<int> values(retained.size());
    thrust::copy(retained.begin(), retained.end(), values.begin());
    require(values == std::vector<int>(33, 9),
            "scratch vector lost storage after its session finished");

    // Exercise stream ordering directly: vector operations may synchronize
    // internally and would otherwise hide an incorrectly ordered free/reuse.
    gpart::detail::ScratchPoolSession ordered;
    gpart::detail::ScratchAllocator<int> allocator;
    constexpr std::size_t count = 257;
    auto source = allocator.allocate(count);
    auto saved = allocator.allocate(count);
    CUDA_CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(source), 0x11,
                              count * sizeof(int), nullptr));
    CUDA_CHECK(cudaMemcpyAsync(thrust::raw_pointer_cast(saved),
        thrust::raw_pointer_cast(source), count * sizeof(int),
        cudaMemcpyDeviceToDevice, nullptr));
    allocator.deallocate(source, count);
    auto reused = allocator.allocate(count);
    CUDA_CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(reused), 0x22,
                              count * sizeof(int), nullptr));
    std::vector<int> saved_values(count);
    CUDA_CHECK(cudaMemcpy(saved_values.data(), thrust::raw_pointer_cast(saved),
                         count * sizeof(int), cudaMemcpyDeviceToHost));
    require(saved_values == std::vector<int>(count, 0x11111111),
            "scratch reuse overtook a pending default-stream read");
    allocator.deallocate(saved, count);
    allocator.deallocate(reused, count);
    ordered.finish();
}

template <typename Types>
void check_parallel_cut() {
    using V = typename Types::VertexT;
    using W = typename Types::WeightT;
    constexpr int n = 259;
    std::vector<std::vector<V>> rows(n);
    const auto add = [&](int v, int u) { rows[v].push_back(u); rows[u].push_back(v); };
    for (int v = 1; v < n - 1; ++v) {
        add(0, v);
        add(0, v); // Repeated entries must each contribute their weight.
        if (v > 1) add(v - 1, v);
    }
    gpart::WeightedGraph<Types> graph;
    graph.offsets.push_back(0);
    graph.vertex_weights.assign(n, 1);
    W weight = 7;
    if constexpr (sizeof(W) == 8) weight = W{1} << 40;
    for (auto& row : rows) {
        std::sort(row.begin(), row.end());
        graph.neighbors.insert(graph.neighbors.end(), row.begin(), row.end());
        graph.edge_weights.insert(graph.edge_weights.end(), row.size(), weight);
        graph.offsets.push_back(graph.neighbors.size());
    }
    // Hub, isolated tail vertex, partial final block and >32-bit cut totals.
    for (bool interleaved : {false, true}) {
        std::vector<V> labels(n);
        for (int v = 0; v < n; ++v) labels[v] = interleaved ? v % 4 : v / 65;
        const auto original = labels;
        const auto expected = gpart::host_cut(graph, labels);
        gpart::RefineOptions options;
        options.parts = 4;
        options.imbalance_ratio = 1.10;
        options.max_rounds = 0;
        options.strict_verify = true;
        gpart::RefineStats stats;
        gpart::refine_partition(graph, labels, options, &stats);
        require(labels == original && stats.initial_cut == expected && stats.final_cut == expected,
                "parallel cut differs from independent CPU cut");
    }
}

template <typename Types>
void check_cross_edge_contraction() {
    using V = typename Types::VertexT;
    using W = typename Types::WeightT;
    using O = typename Types::OffsetT;
    // Non-warp-aligned row count, empty rows, duplicate entries, self loops,
    // all-internal edges, and repeated calls using the same scratch buffers.
    gpart::SclpWorkspace<Types> workspace;
    for (int groups : {7, 1, 7}) {
        constexpr int n = 35;
        gpart::WeightedGraph<Types> input, expected;
        std::vector<V> mapping(n);
        expected.vertex_weights.assign(groups, W{0});
        input.vertex_weights.assign(n, W{1});
        input.offsets.push_back(0);
        std::map<std::pair<V, V>, W> edges;
        W weight = W{3};
        if constexpr (sizeof(W) == 8) weight = W{1} << 40;
        for (int v = 0; v < n; ++v) {
            mapping[v] = v % groups;
            ++expected.vertex_weights[mapping[v]];
            for (int u = 0; u < n - 1 && v < n - 1; ++u) {
                for (int duplicate = 0; duplicate < 2; ++duplicate) {
                    input.neighbors.push_back(u);
                    input.edge_weights.push_back(weight);
                    if (v % groups != u % groups)
                        edges[{v % groups, u % groups}] += weight;
                }
            }
            input.offsets.push_back(static_cast<O>(input.neighbors.size()));
        }
        expected.offsets.push_back(0);
        for (int v = 0; v < groups; ++v) {
            for (const auto& edge : edges) if (edge.first.first == v) {
                expected.neighbors.push_back(edge.first.second);
                expected.edge_weights.push_back(edge.second);
            }
            expected.offsets.push_back(static_cast<O>(expected.neighbors.size()));
        }
        gpart::DeviceAggregateResult<Types> aggregate;
        aggregate.map.assign(mapping.begin(), mapping.end());
        aggregate.vertex_weights.assign(expected.vertex_weights.begin(), expected.vertex_weights.end());
        aggregate.coarse_vertices = groups;
        gpart::ContractionTimings timing;
        auto actual = gpart::copy_device_weighted(gpart::contract(
            gpart::make_device_weighted(input), std::move(aggregate), workspace, timing));
        require(actual.offsets == expected.offsets && actual.neighbors == expected.neighbors &&
                actual.edge_weights == expected.edge_weights && actual.vertex_weights == expected.vertex_weights,
                "cross-edge contraction differs from independent CPU reduction");
    }
    for (bool empty : {false, true}) {
        auto input = make_hierarchy<Types>().levels.front();
        if (empty) {
            input.neighbors.clear();
            input.edge_weights.clear();
            input.offsets.assign(input.vertices() + 1, 0);
        }
        std::vector<V> identity{0, 1, 2, 3};
        gpart::DeviceAggregateResult<Types> aggregate;
        aggregate.coarse_vertices = 4;
        aggregate.map.assign(identity.begin(), identity.end());
        aggregate.vertex_weights.assign(input.vertex_weights.begin(), input.vertex_weights.end());
        gpart::ContractionTimings timing;
        auto actual = gpart::copy_device_weighted(gpart::contract(
            gpart::make_device_weighted(input), std::move(aggregate), workspace, timing));
        require(actual.offsets == input.offsets && actual.neighbors == input.neighbors &&
                actual.edge_weights == input.edge_weights && actual.vertex_weights == input.vertex_weights,
                "identity contraction changed singleton or empty CSR rows");
    }
}

template <typename Types>
void check_sort_weight_width() {
    using V = typename Types::VertexT;
    using W = typename Types::WeightT;
    // Exercise the hub-sort path, duplicate edges, and the >UINT32_MAX
    // total-weight fallback. Compare every level to the original 64-bit path.
    std::vector<W> tested_weights{W{1}, W{1000000}};
    if constexpr (sizeof(W) == 8) tested_weights.push_back(W{1} << 40);
    for (W weight : tested_weights) {
        constexpr int n = 4096;
        std::vector<std::vector<V>> rows(n);
        const auto add = [&](int a, int b) { rows[a].push_back(b); rows[b].push_back(a); };
        for (int v = 0; v < n; ++v) add(v, (v + 1) % n);
        for (int v = 1; v <= 512; ++v) add(0, v);
        rows[0].push_back(0); // Hub sentinel must not collide with a packed key.
        gpart::WeightedGraph<Types> input;
        input.offsets.push_back(0);
        input.vertex_weights.assign(n, 1);
        for (auto& row : rows) {
            std::sort(row.begin(), row.end());
            input.neighbors.insert(input.neighbors.end(), row.begin(), row.end());
            input.edge_weights.insert(input.edge_weights.end(), row.size(), weight);
            input.offsets.push_back(input.neighbors.size());
        }
        gpart::CoarsenOptions options;
        auto candidate = gpart::coarsen(gpart::make_device_weighted(input), options);
        require(candidate.levels.capacity() >= static_cast<std::size_t>(options.max_levels) + 1 &&
                candidate.fine_to_coarse.capacity() >= static_cast<std::size_t>(options.max_levels),
                "device hierarchy growth can deep-copy previous GPU levels");
        auto reference = gpart::make_device_weighted(input);
        gpart::SclpWorkspace<Types> workspace; // Default standalone path is 64-bit.
        std::size_t produced = 1;
        for (int level = 0; level < options.max_levels && reference.vertices() > 1024; ++level) {
            gpart::SclpStats stats;
            auto aggregate = gpart::aggregate(reference, workspace, 4, level, level, stats, false);
            if (aggregate.coarse_vertices >= reference.vertices() ||
                double(aggregate.coarse_vertices) / reference.vertices() > options.stop_contraction_ratio) break;
            require(produced < candidate.levels.size(), "packed hierarchy ended early");
            std::vector<V> expected_map(aggregate.map.size()), actual_map(aggregate.map.size());
            thrust::copy(aggregate.map.begin(), aggregate.map.end(), expected_map.begin());
            thrust::copy(candidate.fine_to_coarse[level].begin(), candidate.fine_to_coarse[level].end(), actual_map.begin());
            require(expected_map == actual_map, "packed affinity changed mapping");
            gpart::ContractionTimings timing;
            reference = gpart::contract(std::move(reference), std::move(aggregate), workspace, timing);
            auto expected = gpart::copy_device_weighted(reference);
            auto actual = gpart::copy_device_weighted(candidate.levels[produced++]);
            require(expected.offsets == actual.offsets && expected.neighbors == actual.neighbors &&
                    expected.edge_weights == actual.edge_weights && expected.vertex_weights == actual.vertex_weights,
                    "packed sort/reduce changed CSR");
        }
        require(produced == candidate.levels.size(), "packed hierarchy has extra levels");
    }
}

}  // namespace

int main() {
    try {
        check_scratch_pool_lifetime();
        gpart::detail::ScratchPoolSession scratch_pool;
        check_scratch_pool_lifetime();
        using Types = gpart::ActiveTypes;
        using VertexT = typename Types::VertexT;
        check_parallel_cut<Types>();
        check_cross_edge_contraction<Types>();
        check_sort_weight_width<Types>();
        auto hierarchy = make_hierarchy<Types>();
        // Both CSR validation paths must retain all rejection conditions.
        for (int failure = 0; failure < 4; ++failure) {
            auto invalid = hierarchy.levels.front();
            if (failure == 0) invalid.neighbors[0] = -1;
            if (failure == 1) invalid.neighbors[0] = invalid.vertices();
            if (failure == 2) invalid.offsets[2] = 0;
            if (failure == 3) invalid.neighbors[0] = 0;
            bool rejected = false;
            try { gpart::validate_weighted_shape(invalid, failure != 3); }
            catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "CSR safety validation was lost");
            if (failure == 3) gpart::validate_weighted_shape(invalid, true);
        }
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
        auto production_options = options;
        production_options.strict_verify = false;
        require(
            gpart::uncoarsen(hierarchy, coarse_partition, production_options) ==
                final_partition,
            "production and strict partitions differ");
        for (bool strict : {false, true}) {
            gpart::DeviceHierarchy<Types> device_hierarchy;
            for (const auto& level : hierarchy.levels)
                device_hierarchy.levels.push_back(gpart::make_device_weighted(level));
            for (const auto& mapping : hierarchy.fine_to_coarse)
                device_hierarchy.fine_to_coarse.emplace_back(mapping);
            thrust::device_vector<VertexT> device_labels = coarse_partition;
            auto resident_options = options;
            resident_options.strict_verify = strict;
            auto result = gpart::uncoarsen(device_hierarchy, std::move(device_labels), resident_options);
            std::vector<VertexT> labels(result.device_partition.size());
            thrust::copy(result.device_partition.begin(), result.device_partition.end(), labels.begin());
            require(labels == final_partition, "device hierarchy changed refinement");
            require(result.partition.empty(), "resident path downloaded partition");
            for (const auto& level : result.levels)
                require(level.timings.graph_h2d_seconds == 0, "resident level uploaded graph");
        }

        // Exact level/map comparison, including the no-contraction case.
        for (int n : {8, 4096}) {
            gpart::WeightedGraph<Types> input;
            input.vertex_weights.assign(n, 1);
            input.offsets.push_back(0);
            for (int v = 0; v < n; ++v) {
                input.neighbors.push_back(v ^ 1);
                input.edge_weights.push_back(1);
                input.offsets.push_back(v + 1);
            }
            gpart::CoarsenOptions co;
            co.strict_verify = true;
            auto host_levels = gpart::coarsen(input, co);
            for (bool strict : {false, true}) {
                co.strict_verify = strict;
                auto device_levels = gpart::coarsen(gpart::make_device_weighted(input), co);
                require(host_levels.levels.size() == device_levels.levels.size(), "resident level count differs");
                require(host_levels.stop_reason == device_levels.stop_reason, "resident stop reason differs");
                if (!strict) require(device_levels.snapshot_seconds == 0, "production copied snapshots");
                for (std::size_t l = 0; l < host_levels.levels.size(); ++l) {
                    auto actual = gpart::copy_device_weighted(device_levels.levels[l]);
                    const auto& expected = host_levels.levels[l];
                    require(actual.offsets == expected.offsets && actual.neighbors == expected.neighbors &&
                            actual.edge_weights == expected.edge_weights && actual.vertex_weights == expected.vertex_weights,
                            "resident hierarchy CSR differs");
                    if (l < host_levels.fine_to_coarse.size()) {
                        std::vector<VertexT> map(device_levels.fine_to_coarse[l].size());
                        thrust::copy(device_levels.fine_to_coarse[l].begin(), device_levels.fine_to_coarse[l].end(), map.begin());
                        require(map == host_levels.fine_to_coarse[l], "resident hierarchy map differs");
                    }
                }
            }
        }
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

        auto pair_graph = make_pair_escape_graph<Types>();
        std::vector<VertexT> pair_partition{0, 0, 1, 1, 0, 0};
        const auto pair_cut_before = gpart::host_cut(
            pair_graph, pair_partition);
        require(pair_cut_before == 4, "pair test initial cut is wrong");
        gpart::RefineStats plain_stats;
        gpart::refine_partition(
            pair_graph, pair_partition, options, &plain_stats);
        require(
            plain_stats.proposals == 0 &&
            gpart::host_cut(pair_graph, pair_partition) == pair_cut_before,
            "plain LP unexpectedly escaped the pair local optimum");

        gpart::PairRefineStats pair_stats;
        gpart::coordinated_pair_escape(
            pair_graph, pair_partition, options, 0, &pair_stats);
        require(pair_stats.candidates == 2, "pair candidate count is wrong");
        require(pair_stats.mutual_pairs == 2, "mutual pair count is wrong");
        require(pair_stats.accepted == 1, "pair admission count is wrong");
        require(!pair_stats.rollback, "improving pair round rolled back");
        require(
            pair_stats.cut_before == 4 && pair_stats.cut_after == 0,
            "pair cut statistics are wrong");
        require(
            pair_partition == std::vector<VertexT>({1, 1, 1, 1, 0, 0}),
            "pair was not committed atomically or pairs overlapped");
        const auto pair_cut_after = gpart::host_cut(
            pair_graph, pair_partition);
        require(
            pair_cut_after < pair_cut_before && pair_cut_after == 0,
            "pair escape did not strictly lower cut");
        std::uint64_t pair_weights[2] = {0, 0};
        for (std::size_t v = 0; v < pair_partition.size(); ++v) {
            pair_weights[pair_partition[v]] += pair_graph.vertex_weights[v];
        }
        require(
            std::max(pair_weights[0], pair_weights[1]) <= 4,
            "pair escape violated balance");
        auto cleanup_options = options;
        cleanup_options.max_rounds = 1;
        gpart::refine_partition(
            pair_graph, pair_partition, cleanup_options, nullptr);
        require(
            gpart::host_cut(pair_graph, pair_partition) == pair_cut_after,
            "one-round cleanup increased the pair result cut");
        std::cout << "uncoarsen_small_ok=1 graph_type="
                  << gpart::kActiveGraphName << " final_cut=" << final_cut
                  << " pair_escape_cut=" << pair_cut_after
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_uncoarsen: " << error.what() << '\n';
        return 1;
    }
}
