#include "graph.hpp"
#include "partitioner.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " <indptr.bin> <indices.bin> <parts> <output.parts>"
              << " [grow_rounds] [refine_rounds] [seeds_per_part]"
              << " [max_vertex_ratio]\n";
}

}

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 9) {
            usage(argv[0]);
            return 2;
        }
        SingleGPUConfig config;
        if (const char* value = std::getenv("INITIAL_PARTITION")) config.initial_partition = value;
        if (const char* value = std::getenv("SEARCH_SEED")) config.search_seed = std::stoi(value);
        if (const char* value = std::getenv("SEED_DISTANCE_POWER")) config.seed_distance_power = std::stoi(value);
        if(config.seed_distance_power<1||config.seed_distance_power>4)throw std::runtime_error("SEED_DISTANCE_POWER must be in 1..4");
        if (const char* value = std::getenv("RESTORE_BEST_CYCLE")) config.restore_best_cycle = std::string(value) != "0";
        config.parts = std::stoi(argv[3]);
        if (argc >= 6) config.grow_rounds = std::stoi(argv[5]);
        if (argc >= 7) config.refine_rounds = std::stoi(argv[6]);
        if (argc >= 8) config.seeds_per_part = std::stoi(argv[7]);
        if (const char* value = std::getenv("MAX_VERTEX_RATIO")) {
            config.vertex_ratio = std::stof(value);
        }
        if (const char* value = std::getenv("DESCENT_MICRO_BATCH")) {
            config.descent_micro_batch = std::stoi(value);
        }
        if (const char* value = std::getenv("DESCENT_MICRO_ROUNDS")) {
            config.descent_micro_rounds = std::stoi(value);
        }
        if (argc >= 9) config.vertex_ratio = std::stof(argv[8]);
        if (const char* value = std::getenv("GLOBAL_CYCLES")) {
            config.global_cycles = std::stoi(value);
        }
        if (const char* value = std::getenv("FIELD_ROUNDS")) {
            config.field_rounds = std::stoi(value);
        }
        if (const char* value = std::getenv("ENABLE_FIELD")) {
            config.enable_field = std::string(value) != "0";
        }
        if (const char* value = std::getenv("TIME_BUDGET_SECONDS")) {
            config.time_budget_seconds = std::stod(value);
        }
        if (const char* value = std::getenv("BALANCE_ROUNDS")) {
            config.balance_rounds = std::stoi(value);
        }
        if (const char* value = std::getenv("POLISH_ROUNDS")) {
            config.polish_rounds = std::stoi(value);
        }
        if (const char* value = std::getenv("FEASIBLE_RECORDER")) {
            config.feasible_recorder = std::string(value) != "0";
        }
        if (const char* value = std::getenv("OSCILLATION_GUARD")) {
            config.oscillation_guard = std::string(value) != "0";
        }
        if (const char* value = std::getenv("MINIMAL_BALANCE_REPAIR")) {
            config.minimal_balance_repair = std::string(value) != "0";
        }
        if (const char* value = std::getenv("INCREMENTAL_CUT")) {
            config.incremental_cut = std::string(value) != "0";
        }
        if (const char* value = std::getenv("INCREMENTAL_CUT_VERIFY")) {
            config.incremental_cut_verify = std::string(value) != "0";
        }
        if (const char* value = std::getenv("INCREMENTAL_CUT_MAX_MOVED_RATIO")) {
            config.incremental_cut_max_moved_ratio = std::stof(value);
        }
        if (const char* value = std::getenv("CACHED_NEIGHBOR_COUNTS")) {
            config.cached_neighbor_counts = std::string(value) != "0";
        }
        if (const char* value = std::getenv("NEIGHBOR_COUNT_DELTA_MAX_MOVED_RATIO")) {
            config.neighbor_count_delta_max_moved_ratio = std::stof(value);
        }
        if (const char* value = std::getenv("STRUCTURAL_WARP_DEGREE")) {
            config.structural_warp_degree = std::stoi(value);
        }
        if (const char* value = std::getenv("PAIR_EXCHANGE")) {
            config.pair_exchange = std::string(value) != "0";
        }
        if (const char* value = std::getenv("PAIR_EXCHANGE_ROUNDS")) {
            config.pair_exchange_rounds = std::stoi(value);
        }
        if (const char* value = std::getenv("PAIR_TOP_TARGETS")) {
            config.pair_top_targets = std::stoi(value);
        }
        if (const char* value = std::getenv("PAIR_BUCKET_LIMIT")) {
            config.pair_bucket_limit = std::stoi(value);
        }
        if (const char* value = std::getenv("PAIR_VERIFY")) {
            config.pair_verify = std::string(value) != "0";
        }
        if (const char* value = std::getenv("BLOCK_LP")) {
            config.block_lp = std::string(value) != "0";
        }
        if (const char* value = std::getenv("BLOCK_MAX_SIZE")) {
            config.block_max_size = std::stoi(value);
        }
        if (const char* value = std::getenv("BLOCK_SEEDS_PER_PAIR")) {
            config.block_seeds_per_pair = std::stoi(value);
        }
        if (const char* value = std::getenv("BLOCK_FRONTIER_LIMIT")) {
            config.block_frontier_limit = std::stoi(value);
        }
        if (const char* value = std::getenv("BLOCK_ROUNDS")) {
            config.block_rounds = std::stoi(value);
        }
        if (const char* value = std::getenv("BLOCK_VERIFY")) {
            config.block_verify = std::string(value) != "0";
        }

        std::cout << "parts=" << config.parts
                  << " search_seed=" << config.search_seed
                  << " seed_distance_power=" << config.seed_distance_power
                  << " restore_best_cycle=" << config.restore_best_cycle
                  << " initial_partition=" << (config.initial_partition.empty()?"generated":config.initial_partition)
                  << " max_vertex_ratio=" << config.vertex_ratio
                  << " grow_rounds=" << config.grow_rounds
                  << " refine_rounds=" << config.refine_rounds
                  << " seeds_per_part=" << config.seeds_per_part
                  << " descent_micro_batch=" << config.descent_micro_batch
                  << " descent_micro_rounds=" << config.descent_micro_rounds
                  << " global_cycles=" << config.global_cycles
                  << " field_rounds=" << config.field_rounds
                  << " enable_field=" << (config.enable_field ? 1 : 0)
                  << " time_budget_seconds=" << config.time_budget_seconds
                  << " balance_rounds=" << config.balance_rounds
                  << " polish_rounds=" << config.polish_rounds
                  << " feasible_recorder=" << (config.feasible_recorder ? 1 : 0)
                  << " oscillation_guard=" << (config.oscillation_guard ? 1 : 0)
                  << " minimal_balance_repair="
                  << (config.minimal_balance_repair ? 1 : 0)
                  << " incremental_cut=" << (config.incremental_cut ? 1 : 0)
                  << " incremental_cut_verify="
                  << (config.incremental_cut_verify ? 1 : 0)
                  << " incremental_cut_max_moved_ratio="
                  << config.incremental_cut_max_moved_ratio
                  << " cached_neighbor_counts="
                  << (config.cached_neighbor_counts ? 1 : 0)
                  << " neighbor_count_delta_max_moved_ratio="
                  << config.neighbor_count_delta_max_moved_ratio
                  << " structural_warp_degree="
                  << config.structural_warp_degree
                  << " pair_exchange=" << (config.pair_exchange ? 1 : 0)
                  << " pair_exchange_rounds=" << config.pair_exchange_rounds
                  << " pair_top_targets=" << config.pair_top_targets
                  << " pair_bucket_limit=" << config.pair_bucket_limit
                  << " pair_verify=" << (config.pair_verify ? 1 : 0)
                  << " block_lp=" << (config.block_lp ? 1 : 0)
                  << " block_max_size=" << config.block_max_size
                  << " block_seeds_per_pair=" << config.block_seeds_per_pair
                  << " block_frontier_limit=" << config.block_frontier_limit
                  << " block_rounds=" << config.block_rounds
                  << " block_verify=" << (config.block_verify ? 1 : 0)
                  << " algorithm=label_propagation_structural_field\n";

        CSRGraph graph;
        graph.load(argv[1], argv[2]);
        SingleGPUPartitioner partitioner(graph, config);
        partitioner.run();
        partitioner.save(argv[4]);
        std::cout << "saved=" << argv[4] << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
