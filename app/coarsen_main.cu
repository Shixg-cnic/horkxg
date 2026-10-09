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
        int positional_argc = argc;
        const bool verify = argc > 1 && std::string(argv[argc - 1]) == "--verify";
        if (verify) --positional_argc;
        if (positional_argc < 5 || positional_argc > 8) {
            std::cerr << "Usage: " << argv[0]
                      << " <indptr.bin> <indices.bin> <parts> <hierarchy.out>"
                      << " [seed=0] [stop_contraction_ratio=0.90] [max_levels=24] [--verify]\n";
            return 2;
        }
        const int parts = std::stoi(argv[3]);
        const std::uint32_t seed = positional_argc >= 6
            ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 0U;
        const double stop_contraction_ratio = positional_argc >= 7
            ? std::stod(argv[6]) : 0.90;
        const int max_levels = positional_argc >= 8 ? std::stoi(argv[7]) : 24;
        if (parts < 2 || parts > 32) {
            throw std::runtime_error("parts must be between 2 and 32");
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
        options.strict_verify = verify;
        auto hierarchy = gpart::coarsen<Types>(input, options);

        const bool skip_export =
            std::getenv("ML_SKIP_HIERARCHY_EXPORT") != nullptr;
        const auto export_start = std::chrono::steady_clock::now();
        if (!skip_export) {
            gpart::write_hierarchy(hierarchy, argv[4]);
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
