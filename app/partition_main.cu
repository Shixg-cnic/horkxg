#include "coarsen.hpp"
#include "graph.hpp"
#include "graph_types.hpp"
#include "initial_partition.hpp"
#include "uncoarsen.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 7) {
            std::cerr << "Usage: " << argv[0]
                      << " <indptr.bin> <indices.bin> <parts> <partition.out>"
                      << " [imbalance_ratio] [seed]\n";
            return 2;
        }
        using Types = gpart::ActiveTypes;
        const int parts = std::stoi(argv[3]);
        const double imbalance = argc >= 6 ? std::stod(argv[5]) : 1.10;
        const auto seed = argc >= 7
            ? static_cast<std::uint32_t>(std::stoul(argv[6])) : 0U;
        auto graph = gpart::load_weighted_graph<Types>(argv[1], argv[2]);
        gpart::CoarsenOptions coarsen_options;
        coarsen_options.parts = parts;
        coarsen_options.seed = seed;
        coarsen_options.strict_verify = true;
        auto hierarchy = gpart::coarsen<Types>(graph, coarsen_options);
        auto coarse_partition = gpart::initial_partition<Types>(
            hierarchy.levels.back(), parts, imbalance);
        gpart::RefineOptions refine_options;
        refine_options.parts = parts;
        refine_options.imbalance_ratio = imbalance;
        refine_options.seed = seed;
        refine_options.max_rounds = 4;
        refine_options.strict_verify = true;
        const auto partition = gpart::uncoarsen<Types>(
            hierarchy, coarse_partition, refine_options);

        std::ofstream output(argv[4], std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open partition output");
        output.write(
            reinterpret_cast<const char*>(partition.data()),
            static_cast<std::streamsize>(
                partition.size() * sizeof(typename Types::VertexT)));
        if (!output) throw std::runtime_error("cannot write partition output");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gpart_partition: " << error.what() << '\n';
        return 1;
    }
}
