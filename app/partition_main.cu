#include "coarsen.hpp"
#include "graph.hpp"
#include "graph_types.hpp"
#include "initial_partition.hpp"
#include "uncoarsen.hpp"

#include <cstdint>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        const auto wall_start = std::chrono::steady_clock::now();
        const bool verify = argc > 1 && std::string(argv[argc - 1]) == "--verify";
        const int positional_argc = argc - (verify ? 1 : 0);
        if (positional_argc < 5 || positional_argc > 7) {
            std::cerr << "Usage: " << argv[0]
                      << " <indptr.bin> <indices.bin> <parts> <partition.out>"
                      << " [imbalance_ratio] [seed] [--verify]\n";
            return 2;
        }
        using Types = gpart::ActiveTypes;
        const int parts = std::stoi(argv[3]);
        const double imbalance = positional_argc >= 6 ? std::stod(argv[5]) : 1.10;
        const auto seed = positional_argc >= 7
            ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 0U;
        const auto load_start = std::chrono::steady_clock::now();
        auto graph = gpart::load_weighted_graph<Types>(argv[1], argv[2]);
        const auto load_end = std::chrono::steady_clock::now();
        std::cout << "verify=" << (verify ? 1 : 0) << '\n';
        gpart::CoarsenOptions coarsen_options;
        coarsen_options.parts = parts;
        coarsen_options.seed = seed;
        coarsen_options.strict_verify = verify;
        const auto coarsen_start = std::chrono::steady_clock::now();
        auto hierarchy = gpart::coarsen<Types>(graph, coarsen_options);
        const auto coarsen_end = std::chrono::steady_clock::now();
        const auto initial_start = std::chrono::steady_clock::now();
        auto coarse_partition = gpart::initial_partition<Types>(
            hierarchy.levels.back(), parts, imbalance);
        const auto initial_end = std::chrono::steady_clock::now();
        gpart::RefineOptions refine_options;
        refine_options.parts = parts;
        refine_options.imbalance_ratio = imbalance;
        refine_options.seed = seed;
        refine_options.max_rounds = 4;
        refine_options.strict_verify = verify;
        const auto uncoarsen_start = std::chrono::steady_clock::now();
        const auto partition = gpart::uncoarsen<Types>(
            hierarchy, coarse_partition, refine_options);
        const auto uncoarsen_end = std::chrono::steady_clock::now();

        const auto output_start = std::chrono::steady_clock::now();
        std::ofstream output(argv[4], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open partition output");
        output.write(
            reinterpret_cast<const char*>(partition.data()),
            static_cast<std::streamsize>(
                partition.size() * sizeof(typename Types::VertexT)));
        output.close();
        if (!output) throw std::runtime_error("cannot write partition output");
        const auto output_end = std::chrono::steady_clock::now();
        const double input_seconds = std::chrono::duration<double>(
            load_end - load_start).count();
        const double coarsen_seconds = std::chrono::duration<double>(
            coarsen_end - coarsen_start).count();
        const double initial_seconds = std::chrono::duration<double>(
            initial_end - initial_start).count();
        const double uncoarsen_seconds = std::chrono::duration<double>(
            uncoarsen_end - uncoarsen_start).count();
        const double output_seconds = std::chrono::duration<double>(
            output_end - output_start).count();
        const double algorithm_seconds =
            coarsen_seconds + initial_seconds + uncoarsen_seconds;
        // Host wall-clock stage times include each stage's existing logging.
        // partition_total is an alias of algorithm_total (excludes file I/O).
        std::cout << std::setprecision(10)
                  << "partition_timing input_load_seconds=" << input_seconds
                  << " coarsen_seconds=" << coarsen_seconds
                  << " initial_partition_seconds=" << initial_seconds
                  << " uncoarsen_seconds=" << uncoarsen_seconds
                  << " partition_output_seconds=" << output_seconds
                  << " partition_total_seconds=" << algorithm_seconds
                  << " algorithm_total_seconds=" << algorithm_seconds
                  << " wall_total_seconds="
                  << std::chrono::duration<double>(output_end - wall_start).count()
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gpart_partition: " << error.what() << '\n';
        return 1;
    }
}
