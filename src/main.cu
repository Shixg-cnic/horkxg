#include "check.hpp"
#include "graph.hpp"
#include "sclp.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <thrust/copy.h>

namespace {

using sclp::WeightedGraph;

std::uint64_t host_cut(
    const WeightedGraph& graph, const std::vector<std::int32_t>& labels) {
    std::uint64_t directed = 0;
    for (std::int64_t v = 0; v < graph.vertices(); ++v) {
        for (auto e = graph.offsets[static_cast<std::size_t>(v)];
             e < graph.offsets[static_cast<std::size_t>(v + 1)]; ++e) {
            if (labels[static_cast<std::size_t>(v)] !=
                labels[static_cast<std::size_t>(
                    graph.neighbors[static_cast<std::size_t>(e)])]) {
                directed += graph.edge_weights[static_cast<std::size_t>(e)];
            }
        }
    }
    if (directed & 1ULL) {
        throw std::runtime_error("weighted CSR cut is not symmetric");
    }
    return directed / 2;
}

void validate_weighted_shape(
    const WeightedGraph& graph, bool allow_self_loops) {
    if (graph.offsets.size() != graph.vertex_weights.size() + 1 ||
        graph.offsets.empty() || graph.offsets.front() != 0 ||
        graph.offsets.back() != static_cast<std::int64_t>(graph.neighbors.size()) ||
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

void validate_weighted_csr(
    const WeightedGraph& graph, bool allow_self_loops) {
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
                static_cast<std::int32_t>(v));
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

void validate_coarsening_step(
    const WeightedGraph& fine, const WeightedGraph& coarse,
    const std::vector<std::int32_t>& map, std::uint64_t cluster_cap,
    int level, bool strict_verify) {
    if (map.size() != static_cast<std::size_t>(fine.vertices())) {
        throw std::runtime_error("coarsening map has the wrong length");
    }
    if (coarse.vertices() <= 0 || coarse.vertices() > fine.vertices()) {
        throw std::runtime_error("coarsening produced an invalid vertex count");
    }
    std::vector<std::uint64_t> recomputed(
        static_cast<std::size_t>(coarse.vertices()), 0);
    std::vector<std::uint64_t> coverage(
        static_cast<std::size_t>(coarse.vertices()), 0);
    for (std::int64_t v = 0; v < fine.vertices(); ++v) {
        const auto c = map[static_cast<std::size_t>(v)];
        if (c < 0 || c >= coarse.vertices()) {
            throw std::runtime_error("coarsening map contains an out-of-range id");
        }
        recomputed[static_cast<std::size_t>(c)] +=
            fine.vertex_weights[static_cast<std::size_t>(v)];
        ++coverage[static_cast<std::size_t>(c)];
    }
    const auto fine_total = std::accumulate(
        fine.vertex_weights.begin(), fine.vertex_weights.end(), std::uint64_t{0});
    const auto coarse_total = std::accumulate(
        coarse.vertex_weights.begin(), coarse.vertex_weights.end(), std::uint64_t{0});
    if (fine_total != coarse_total) {
        throw std::runtime_error("coarsening did not conserve vertex weight");
    }
    for (std::int64_t c = 0; c < coarse.vertices(); ++c) {
        if (coverage[static_cast<std::size_t>(c)] == 0 ||
            recomputed[static_cast<std::size_t>(c)] !=
                coarse.vertex_weights[static_cast<std::size_t>(c)]) {
            throw std::runtime_error("coarsening cluster weight mismatch");
        }
        if (coarse.vertex_weights[static_cast<std::size_t>(c)] > cluster_cap) {
            throw std::runtime_error("coarse point exceeds configured weight cap");
        }
    }
    if (strict_verify) {
        validate_weighted_csr(coarse, false);
    } else {
        validate_weighted_shape(coarse, false);
    }

    std::vector<std::int32_t> coarse_labels(
        static_cast<std::size_t>(coarse.vertices()));
    for (std::int64_t c = 0; c < coarse.vertices(); ++c) {
        coarse_labels[static_cast<std::size_t>(c)] =
            static_cast<std::int32_t>((c * 2654435761ULL + 17) % 23);
    }
    std::vector<std::int32_t> fine_labels(map.size());
    for (std::size_t v = 0; v < map.size(); ++v) {
        fine_labels[v] = coarse_labels[static_cast<std::size_t>(map[v])];
    }
    if (host_cut(fine, fine_labels) != host_cut(coarse, coarse_labels)) {
        throw std::runtime_error("cut is not conserved by coarse projection");
    }
    std::cout << "ml_layer_verify level=" << level
              << " map=ok weights=ok csr=ok projection_cut=ok\n";
}

void write_jet_hierarchy(
    const std::vector<WeightedGraph>& levels,
    const std::vector<std::vector<std::int32_t>>& maps,
    const std::string& path) {
    if (levels.empty() || maps.size() + 1 != levels.size()) {
        throw std::runtime_error("cannot write an incomplete Jet hierarchy");
    }
    const auto temporary = path + ".tmp";
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot open hierarchy temporary file");
        }
        const auto level_count = static_cast<std::int32_t>(levels.size());
        output.write(reinterpret_cast<const char*>(&level_count), sizeof(level_count));
        for (std::size_t i = 0; i < levels.size(); ++i) {
            const auto& graph = levels[i];
            if (graph.vertices() > std::numeric_limits<std::int32_t>::max() ||
                graph.edges() > std::numeric_limits<std::int32_t>::max()) {
                throw std::runtime_error(
                    "Jet standard hierarchy requires 32-bit dimensions");
            }
            const auto n = static_cast<std::int32_t>(graph.vertices());
            const auto m = static_cast<std::int32_t>(graph.edges());
            output.write(reinterpret_cast<const char*>(&n), sizeof(n));
            output.write(reinterpret_cast<const char*>(&m), sizeof(m));
            for (const auto offset : graph.offsets) {
                if (offset > std::numeric_limits<std::int32_t>::max()) {
                    throw std::runtime_error("Jet row offset exceeds int32");
                }
                const auto value = static_cast<std::int32_t>(offset);
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            output.write(reinterpret_cast<const char*>(graph.neighbors.data()),
                         static_cast<std::streamsize>(m * sizeof(std::int32_t)));
            for (const auto weight : graph.edge_weights) {
                if (weight > std::numeric_limits<std::int32_t>::max()) {
                    throw std::runtime_error("Jet edge weight exceeds int32");
                }
                const auto value = static_cast<std::int32_t>(weight);
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            for (const auto weight : graph.vertex_weights) {
                if (weight > std::numeric_limits<std::int32_t>::max()) {
                    throw std::runtime_error("Jet vertex weight exceeds int32");
                }
                const auto value = static_cast<std::int32_t>(weight);
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            if (i > 0) {
                const auto& map = maps[i - 1];
                if (map.size() !=
                    static_cast<std::size_t>(levels[i - 1].vertices())) {
                    throw std::runtime_error("Jet map has the wrong fine-level length");
                }
                for (const auto coarse : map) {
                    if (coarse < 0 || coarse >= n) {
                        throw std::runtime_error(
                            "Jet map contains an invalid coarse id");
                    }
                    output.write(
                        reinterpret_cast<const char*>(&coarse), sizeof(coarse));
                }
            }
        }
        output.flush();
        if (!output) throw std::runtime_error("failed while writing Jet hierarchy");
        output.close();
        if (std::rename(temporary.c_str(), path.c_str()) != 0) {
            throw std::runtime_error("cannot commit Jet hierarchy output");
        }
    } catch (...) {
        std::remove(temporary.c_str());
        throw;
    }
}

WeightedGraph make_weighted(const CSRGraph& graph) {
    WeightedGraph out;
    out.offsets = graph.offsets();
    out.neighbors = graph.neighbors();
    out.edge_weights.assign(static_cast<std::size_t>(graph.edges()), 1);
    out.vertex_weights.assign(static_cast<std::size_t>(graph.vertices()), 1);
    return out;
}

#ifdef SCLP_MERGE_DIAGNOSTICS
std::vector<std::int32_t> read_reference_partition(
    const std::string& path, std::int64_t vertices, int parts) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open reference partition: " + path);
    }
    std::vector<std::int32_t> labels;
    labels.reserve(static_cast<std::size_t>(vertices));
    std::int64_t value = 0;
    while (input >> value) {
        if (value < 0 || value >= parts) {
            throw std::runtime_error(
                "reference partition contains an out-of-range label: " + path);
        }
        labels.push_back(static_cast<std::int32_t>(value));
    }
    if (labels.size() != static_cast<std::size_t>(vertices)) {
        throw std::runtime_error(
            "reference partition length does not match graph: " + path);
    }
    return labels;
}

sclp::MergeDiagnosticContext make_merge_diagnostics(
    std::int64_t vertices, int parts) {
    if (parts != sclp::kDiagnosticParts) {
        throw std::runtime_error("merge diagnostics currently require k=4");
    }
    const char* prefix = std::getenv("SCLP_DIAG_PREFIX");
    const char* jet_path = std::getenv("SCLP_DIAG_JET_PART");
    const char* metis_path = std::getenv("SCLP_DIAG_METIS_PART");
    if (prefix == nullptr || jet_path == nullptr || metis_path == nullptr) {
        throw std::runtime_error(
            "diagnostic binary requires SCLP_DIAG_PREFIX, "
            "SCLP_DIAG_JET_PART, and SCLP_DIAG_METIS_PART");
    }
    sclp::MergeDiagnosticContext context;
    context.output_prefix = prefix;
    context.reference_names = {"jet", "metis"};
    const std::array<std::string, 2> paths = {jet_path, metis_path};
    for (int oracle = 0; oracle < 2; ++oracle) {
        const auto labels = read_reference_partition(
            paths[oracle], vertices, parts);
        auto& histograms = context.vertex_histograms[oracle];
        histograms.resize(static_cast<std::size_t>(vertices));
        for (std::int64_t v = 0; v < vertices; ++v) {
            histograms[static_cast<std::size_t>(v)][
                labels[static_cast<std::size_t>(v)]] = 1;
        }
    }
    std::ofstream records(
        context.output_prefix + ".merges.bin",
        std::ios::binary | std::ios::trunc);
    std::ofstream levels(
        context.output_prefix + ".levels.csv", std::ios::trunc);
    if (!records || !levels) {
        throw std::runtime_error("cannot initialize merge diagnostic outputs");
    }
    levels << "role_mode,level,fine_vertices,coarse_vertices,contraction_ratio,"
              "accepted,oracle0,bad0,bad_ratio0,total_loss0,average_loss0,purity0,"
              "oracle1,bad1,bad_ratio1,total_loss1,average_loss1,purity1\n";
    return context;
}
#endif

void run_hierarchy(
    const CSRGraph& input, int parts, std::uint32_t seed,
    double stop_contraction_ratio, int max_levels,
    const std::string& output_path, bool strict_verify) {
    const auto total_start = std::chrono::steady_clock::now();
    const bool diagnostics = std::getenv("SCLP_DIAGNOSTICS") != nullptr;
    const bool verify = strict_verify || std::getenv("SCLP_VERIFY") != nullptr;
    const bool skip_export = std::getenv("ML_SKIP_HIERARCHY_EXPORT") != nullptr;
    std::vector<WeightedGraph> levels{make_weighted(input)};
    std::vector<std::vector<std::int32_t>> maps;
#ifdef SCLP_MERGE_DIAGNOSTICS
    auto merge_diagnostics = make_merge_diagnostics(
        input.vertices(), parts);
#endif

    const auto input_verify_start = std::chrono::steady_clock::now();
    if (strict_verify) {
        validate_weighted_csr(levels.front(), true);
    } else {
        validate_weighted_shape(levels.front(), true);
        std::vector<std::int32_t> labels(
            static_cast<std::size_t>(levels.front().vertices()));
        for (std::int64_t v = 0; v < levels.front().vertices(); ++v) {
            labels[static_cast<std::size_t>(v)] =
                static_cast<std::int32_t>((v * 2654435761ULL + 17) % 23);
        }
        (void)host_cut(levels.front(), labels);
    }
    std::cout << "ml_input_verify_seconds="
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - input_verify_start).count()
              << " status=ok mode=" << (strict_verify ? "strict" : "fast") << '\n';

    const auto device_input_start = std::chrono::steady_clock::now();
    auto current = sclp::make_device_weighted(levels.front());
    sclp::SclpWorkspace workspace;
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "ml_gpu_device_input_seconds="
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - device_input_start).count()
              << " method=sclp\n";

    // With U=ceil(W/(beta*k)), capacity alone keeps the useful coarse scale
    // near beta*k vertices; attempting to drive toward the old 8*k cutoff only
    // produces saturated clusters and an insufficient-contraction stop.
    const std::int64_t cutoff = std::max<std::int64_t>(
        32, sclp::kBeta * static_cast<std::int64_t>(parts));
    const auto core_start = std::chrono::steady_clock::now();
    double device_algorithm_seconds = 0.0;
    double snapshot_seconds = 0.0;
    double affinity_gpu_seconds = 0.0;
    double admission_gpu_seconds = 0.0;
    double two_hop_gpu_seconds = 0.0;
    double compact_gpu_seconds = 0.0;
    double contraction_gpu_seconds = 0.0;
    std::string stop_reason = "capacity_floor";
    int level = 0;
    for (; level < max_levels && current.vertices() > cutoff; ++level) {
        std::cout << "ml_coarsen_begin level=" << level
                  << " vertices=" << current.vertices()
                  << " edges=" << current.edges()
                  << " method=sclp seed="
                  << (seed + static_cast<std::uint32_t>(level)) << '\n';
        sclp::SclpStats stats;
        const auto aggregate_start = std::chrono::steady_clock::now();
        auto aggregate = sclp::aggregate(
            current, workspace, parts, seed + static_cast<std::uint32_t>(level),
            level, stats, diagnostics
#ifdef SCLP_MERGE_DIAGNOSTICS
            , &merge_diagnostics
#endif
            );
        device_algorithm_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - aggregate_start).count();
        const double contraction = static_cast<double>(aggregate.coarse_vertices) /
                                   static_cast<double>(current.vertices());
        affinity_gpu_seconds += stats.affinity_seconds;
        admission_gpu_seconds += stats.admission_seconds;
        two_hop_gpu_seconds += stats.two_hop_seconds;
        compact_gpu_seconds += stats.compact_seconds;
        std::cout << "ml_coarsen_map level=" << level
                  << " coarse_vertices=" << aggregate.coarse_vertices
                  << " ratio=" << contraction
                  << " maximum_weight=" << aggregate.maximum_weight
                  << " capacity=" << aggregate.capacity << '\n';
        if (aggregate.coarse_vertices >= current.vertices() ||
            contraction > stop_contraction_ratio) {
            std::cout << "ml_gpu_timing level=" << level
                      << " affinity_seconds=" << stats.affinity_seconds
                      << " admission_seconds=" << stats.admission_seconds
                      << " two_hop_seconds=" << stats.two_hop_seconds
                      << " compact_seconds=" << stats.compact_seconds
                      << " contraction_seconds=0\n";
            std::cout << "ml_coarsen_stop reason=insufficient_contraction\n";
            stop_reason = "insufficient_contraction";
            break;
        }

        double contract_seconds = 0.0;
        auto coarse_device = sclp::contract(
            current, aggregate, workspace, contract_seconds);
        device_algorithm_seconds += contract_seconds;
        contraction_gpu_seconds += contract_seconds;
        std::cout << "ml_gpu_timing level=" << level
                  << " affinity_seconds=" << stats.affinity_seconds
                  << " admission_seconds=" << stats.admission_seconds
                  << " two_hop_seconds=" << stats.two_hop_seconds
                  << " compact_seconds=" << stats.compact_seconds
                  << " contraction_seconds=" << contract_seconds << '\n';
        std::cout << "ml_gpu_contract_seconds level=" << level
                  << " seconds=" << contract_seconds
                  << " coarse_edges=" << coarse_device.edges() << '\n';

        const auto snapshot_start = std::chrono::steady_clock::now();
        std::vector<std::int32_t> host_map(aggregate.map.size());
        thrust::copy(aggregate.map.begin(), aggregate.map.end(), host_map.begin());
        auto coarse_host = sclp::copy_device_weighted(coarse_device);
        const auto snapshot = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - snapshot_start).count();
        snapshot_seconds += snapshot;
        std::cout << "ml_gpu_device_snapshot_seconds level=" << level
                  << " seconds=" << snapshot << '\n';

        if (verify) {
            const auto verify_start = std::chrono::steady_clock::now();
            validate_coarsening_step(
                levels.back(), coarse_host, host_map, aggregate.capacity,
                level, strict_verify);
            std::cout << "ml_gpu_device_verify_seconds level=" << level
                      << " seconds=" << std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - verify_start).count()
                      << '\n';
        }
        maps.push_back(std::move(host_map));
        levels.push_back(std::move(coarse_host));
        current = std::move(coarse_device);
    }
    if (level == max_levels && current.vertices() > cutoff) {
        stop_reason = "max_levels";
    }
    const auto core_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - core_start).count();
    std::cout << "ml_gpu_device_core_seconds=" << device_algorithm_seconds
              << " aggregate_plus_contract=1 method=sclp\n";
    std::cout << "ml_gpu_timing_total"
              << " affinity_seconds=" << affinity_gpu_seconds
              << " admission_seconds=" << admission_gpu_seconds
              << " two_hop_seconds=" << two_hop_gpu_seconds
              << " compact_seconds=" << compact_gpu_seconds
              << " contraction_seconds=" << contraction_gpu_seconds << '\n';
    std::cout << "ml_hierarchy_loop_seconds=" << core_seconds
              << " includes_snapshot_and_verify=1 method=sclp\n";

    const auto export_start = std::chrono::steady_clock::now();
    if (!skip_export) write_jet_hierarchy(levels, maps, output_path);
    std::cout << "ml_hierarchy_export_seconds="
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - export_start).count()
              << " skipped=" << (skip_export ? 1 : 0) << '\n';
    std::cout << "ml_hierarchy_levels=" << levels.size()
              << " coarsest_vertices=" << levels.back().vertices()
              << " stop_reason=" << stop_reason
              << " stop_contraction_ratio=" << stop_contraction_ratio
              << " method=sclp\n";
    std::cout << "ml_gpu_device_snapshot_total_seconds=" << snapshot_seconds
              << " method=sclp\n";
    std::cout << "ml_total_seconds="
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - total_start).count()
              << " hierarchy=" << (skip_export ? "skipped" : output_path)
              << " coarsen_only=1 method=sclp\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 11) {
            std::cerr << "Usage: " << argv[0]
                      << " <indptr.bin> <indices.bin> <parts> <hierarchy.out>"
                      << " [max_vertex_ratio] [seed] [stop_contraction_ratio]"
                      << " [coarsen_method=sclp] [unused] [max_levels]\n";
            return 2;
        }
        const int parts = std::stoi(argv[3]);
        const double ratio = argc >= 6 ? std::stod(argv[5]) : 1.10;
        const std::uint32_t seed = argc >= 7
            ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 0U;
        const double stop_contraction_ratio = argc >= 8
            ? std::stod(argv[7]) : 0.90;
        const std::string method = argc >= 9 ? argv[8] : "sclp";
        const int max_levels = argc >= 11 ? std::stoi(argv[10]) : 24;
        if (parts < 2 || parts > 32 || ratio < 1.0) {
            throw std::runtime_error("invalid parts or maximum vertex ratio");
        }
        if (method != "sclp") {
            throw std::runtime_error("coarsen_method must be sclp");
        }
        if (stop_contraction_ratio <= 0.0 || stop_contraction_ratio > 1.0) {
            throw std::runtime_error("invalid coarsening stop ratio");
        }
        if (max_levels <= 0 || max_levels > 200) {
            throw std::runtime_error("invalid coarsening max levels");
        }

        CSRGraph input;
        input.load(argv[1], argv[2]);
        const bool strict_verify =
            std::getenv("GPU_LP_STRICT_VERIFY") != nullptr;
        run_hierarchy(
            input, parts, seed, stop_contraction_ratio, max_levels,
            argv[4], strict_verify);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multilevel_lp: " << error.what() << '\n';
        return 1;
    }
}
