#include "coarsen.hpp"
#include "graph.hpp"
#include "graph_types.hpp"
#include "initial_partition.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 6) {
            std::cerr << "Usage: " << argv[0]
                      << " <indptr.bin> <indices.bin> <parts>"
                      << " [imbalance_ratio] [seed]\n";
            return 2;
        }
        using Types = gpart::ActiveTypes;
        const int parts = std::stoi(argv[3]);
        const double imbalance = argc >= 5 ? std::stod(argv[4]) : 1.10;
        const auto seed = argc >= 6
            ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 0U;
        auto graph = gpart::load_weighted_graph<Types>(argv[1], argv[2]);
        gpart::CoarsenOptions options;
        options.parts = parts;
        options.seed = seed;
        auto hierarchy = gpart::coarsen<Types>(graph, options);
        const auto partition = gpart::initial_partition<Types>(
            hierarchy.levels.back(), parts, imbalance);
        if (partition.size() != static_cast<std::size_t>(
                                    hierarchy.levels.back().vertices())) {
            throw std::runtime_error("initial partition has the wrong length");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "gpart_coarsen_initial: " << error.what() << '\n';
        return 1;
    }
}
