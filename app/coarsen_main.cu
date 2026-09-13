#include "coarsen.hpp"
#include "graph.hpp"
#include "graph_types.hpp"
#include "hierarchy.hpp"

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

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

        using Types = gpart::ActiveTypes;
        auto input = gpart::load_weighted_graph<Types>(argv[1], argv[2]);
        const auto total_start = std::chrono::steady_clock::now();
        gpart::CoarsenOptions options;
        options.parts = parts;
        options.seed = seed;
        options.stop_contraction_ratio = stop_contraction_ratio;
        options.max_levels = max_levels;
        options.strict_verify =
            std::getenv("GPU_LP_STRICT_VERIFY") != nullptr;
        auto hierarchy = gpart::coarsen<Types>(input, options);

        const bool skip_export =
            std::getenv("ML_SKIP_HIERARCHY_EXPORT") != nullptr;
        const auto export_start = std::chrono::steady_clock::now();
        if (!skip_export) {
            gpart::write_jet_hierarchy(hierarchy, argv[4]);
        }
        std::cout << "ml_hierarchy_export_seconds="
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - export_start).count()
                  << " skipped=" << (skip_export ? 1 : 0) << '\n';
        std::cout << "ml_hierarchy_levels=" << hierarchy.layer_count()
                  << " coarsest_vertices=" << hierarchy.levels.back().vertices()
                  << " stop_reason=" << hierarchy.stop_reason
                  << " stop_contraction_ratio=" << stop_contraction_ratio
                  << " method=sclp\n";
        std::cout << "ml_gpu_device_snapshot_total_seconds="
                  << hierarchy.snapshot_seconds << " method=sclp\n";
        std::cout << "ml_total_seconds="
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - total_start).count()
                  << " hierarchy=" << (skip_export ? "skipped" : argv[4])
                  << " coarsen_only=1 method=sclp graph_type="
                  << gpart::kActiveGraphName << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gpart_coarsen: " << error.what() << '\n';
        return 1;
    }
}
