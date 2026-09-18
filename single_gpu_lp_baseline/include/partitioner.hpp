#pragma once

#include "graph.hpp"

#include <cstdint>
#include <string>
#include <vector>

struct SingleGPUConfig {
    int parts = 4;
    int grow_rounds = 30;
    int refine_rounds = 50;
    int seeds_per_part = 1;
    int global_cycles = 5;
    int field_rounds = 8;
    bool enable_field = true;
    double time_budget_seconds = 0.0;
    int balance_rounds = 5;
    int polish_rounds = 20;
    float vertex_ratio = 1.10f;
    int min_gain = 1;
    bool repair = true;
    int descent_micro_batch = 0;
    int descent_micro_rounds = 8;
    bool feasible_recorder = true;
    bool oscillation_guard = false;
    bool minimal_balance_repair = false;
    bool incremental_cut = true;
    bool incremental_cut_verify = false;
    float incremental_cut_max_moved_ratio = 0.05f;
    bool cached_neighbor_counts = true;
    float neighbor_count_delta_max_moved_ratio = 0.15f;
    int structural_warp_degree = 64;
    int search_seed = 0;
    int seed_distance_power = 1;
    bool restore_best_cycle = false;
    std::string initial_partition;
    bool pair_exchange = false;
    int pair_exchange_rounds = 2;
    int pair_top_targets = 2;
    int pair_bucket_limit = 32;
    bool pair_verify = false;
    bool block_lp = false;
    int block_max_size = 16;
    int block_seeds_per_pair = 8;
    int block_frontier_limit = 256;
    int block_rounds = 2;
    bool block_verify = false;
};

struct SingleGPUMetrics {
    std::uint64_t cut = 0;
    std::uint64_t edges = 0;
    std::vector<std::uint64_t> vertex_loads;
};

class SingleGPUPartitioner {
public:
    SingleGPUPartitioner(
        const CSRGraph& graph, const SingleGPUConfig& config,
        const std::vector<std::int32_t>* initial_labels = nullptr);
    ~SingleGPUPartitioner();

    SingleGPUPartitioner(const SingleGPUPartitioner&) = delete;
    SingleGPUPartitioner& operator=(const SingleGPUPartitioner&) = delete;

    void run();
    void save(const std::string& output_path);
    SingleGPUMetrics metrics() const;
    const std::vector<std::int32_t>& labels() const { return labels_; }

private:
    struct BlockRoundStats {
        int start_candidates = 0;
        int seed_pool_candidates = 0;
        int positive_candidates = 0;
        int size_ge2_positive = 0;
        int nonpositive_seed_positive = 0;
        int frontier_overflows = 0;
        int overlap_eliminated = 0;
        int capacity_eliminated = 0;
        int selected_candidates = 0;
        int moved_vertices = 0;
        std::int64_t candidate_gain_sum = 0;
        std::int64_t batch_gain = 0;
        std::int64_t best_candidate_gain = 0;
        double median_candidate_gain = 0.0;
        int best_candidate_size = 0;
        double median_candidate_size = 0.0;
        std::uint64_t edge_visits = 0;
        std::uint64_t trial_cut = 0;
        std::uint64_t cut_before = 0;
        std::uint64_t cut_after = 0;
        double seed_seconds = 0.0;
        double growth_seconds = 0.0;
        double selection_seconds = 0.0;
        double trial_seconds = 0.0;
        double cache_rebuild_seconds = 0.0;
        double post_polish_gain = 0.0;
        double seconds = 0.0;
        bool accepted = false;
    };

    void allocate();
    void release() noexcept;
    void build_degree_normalization();
    void choose_distance_separated_seeds();
    void distance_bfs(std::int64_t source);
    void distance_grow();
    void build_current_label_field();
    void build_neighbor_label_counts();
    void update_neighbor_label_counts(int moved, int candidate_count);
    int pair_exchange_round(
        int round, std::vector<std::uint64_t>& loads,
        std::uint64_t current_cut, std::uint64_t& updated_cut,
        std::uint64_t& proposed_exchanges, std::uint64_t& accepted_exchanges,
        std::int64_t& accepted_gain, int& candidate_records,
        int& boundary_vertices, double& candidate_seconds,
        double& filter_seconds, double& cache_rebuild_seconds,
        double& submit_seconds, double& seconds);
    bool block_lp_round(
        int round, std::vector<std::uint64_t>& loads,
        std::uint64_t current_cut, std::uint64_t& updated_cut,
        BlockRoundStats& stats);
    void load_counts(std::vector<std::uint64_t>& loads);
    int refine_round(int round, std::vector<std::uint64_t>& loads,
                     bool field_projection, bool balance_projection,
                     bool conflict_aware = false);
    int refine_round_once(int round, std::vector<std::uint64_t>& loads,
                          bool field_projection, bool balance_projection,
                          bool conflict_aware);
    std::uint64_t compute_cut();
    std::uint64_t compute_cut_for_labels(const std::int32_t* labels);
    std::int64_t compute_cut_delta(int candidate_count, int moved);

    const CSRGraph& graph_;
    SingleGPUConfig config_;
    const std::vector<std::int32_t>* initial_labels_ = nullptr;
    std::int64_t n_ = 0;
    std::int64_t m_ = 0;
    std::int64_t capacity_ = 0;
    std::int64_t active_capacity_ = 0;
    std::int64_t exploration_capacity_ = 0;
    std::int64_t gain_abs_bound_ = 1;
    int last_candidate_count_ = 0;
    int last_proposed_candidate_count_ = 0;
    int last_quota_count_ = 0;
    int last_applied_count_ = 0;
    int last_capacity_rejected_ = 0;
    std::int64_t last_candidate_score_sum_ = 0;
    std::int64_t last_applied_score_sum_ = 0;
    std::int64_t last_refine_cut_delta_ = 0;
    bool last_all_candidates_fit_ = false;
    int high_degree_count_ = 0;
    bool last_moved_vertices_compacted_ = false;

    std::vector<std::int64_t> seeds_;
    std::vector<std::int32_t> labels_;

    std::int64_t* d_offsets_ = nullptr;
    std::int32_t* d_neighbors_ = nullptr;
    std::int32_t* d_labels_ = nullptr;
    std::int32_t* d_next_labels_ = nullptr;
    std::int32_t* d_time_checkpoint_labels_ = nullptr;
    std::int32_t* d_time_checkpoint_best_labels_ = nullptr;
    std::int32_t* d_targets_ = nullptr;
    std::int32_t* d_gains_ = nullptr;
    std::int32_t* d_best_labels_ = nullptr;
    std::int32_t* d_old_labels_ = nullptr;
    std::int32_t* d_neighbor_label_counts_ = nullptr;
    std::uint8_t* d_core_flags_ = nullptr;
    std::int64_t* d_candidates_ = nullptr;
    std::int64_t* d_sorted_candidates_ = nullptr;
    std::int64_t* d_high_degree_vertices_ = nullptr;
    std::uint64_t* d_keys_ = nullptr;
    std::uint64_t* d_sorted_keys_ = nullptr;
    unsigned long long* d_loads_ = nullptr;
    unsigned long long* d_load_deltas_ = nullptr;
    unsigned long long* d_gain_hist_ = nullptr;
    unsigned long long* d_cut_ = nullptr;
    std::uint32_t* d_target_quota_ = nullptr;
    std::uint32_t* d_source_quota_ = nullptr;
    std::uint32_t* d_target_begin_ = nullptr;
    float* d_inv_sqrt_degree_ = nullptr;
    float* d_signal_ = nullptr;
    float* d_next_signal_ = nullptr;
    int* d_candidate_count_ = nullptr;
    int* d_changed_ = nullptr;
    std::uint8_t* d_moved_flags_ = nullptr;
    std::int64_t* d_cut_delta_ = nullptr;
    std::int32_t* d_pair_targets_ = nullptr;
    std::int32_t* d_pair_gains_ = nullptr;
    std::uint64_t* d_pair_keys_ = nullptr;
    std::uint64_t* d_pair_sorted_keys_ = nullptr;
    std::int64_t* d_pair_values_ = nullptr;
    std::int64_t* d_pair_sorted_values_ = nullptr;
    std::int32_t* d_pair_bucket_counts_ = nullptr;
    std::int32_t* d_pair_bucket_begin_ = nullptr;
    std::int64_t* d_pair_u_ = nullptr;
    std::int64_t* d_pair_v_ = nullptr;
    std::int64_t* d_pair_result_gains_ = nullptr;
    std::uint8_t* d_pair_proposed_flags_ = nullptr;
    std::uint8_t* d_pair_accepted_flags_ = nullptr;
    std::int64_t* d_pair_moved_vertices_ = nullptr;
    int* d_pair_moved_count_ = nullptr;
    int* d_pair_boundary_count_ = nullptr;
    int* d_pair_accepted_count_ = nullptr;
    std::int64_t* d_pair_accepted_gain_ = nullptr;
    std::int32_t* d_block_seed_vertices_ = nullptr;
    std::int64_t* d_block_seed_gains_ = nullptr;
    std::int32_t* d_block_seed_sources_ = nullptr;
    std::int32_t* d_block_seed_targets_ = nullptr;
    std::int32_t* d_block_vertices_ = nullptr;
    std::int64_t* d_block_candidate_gains_ = nullptr;
    std::int32_t* d_block_candidate_sizes_ = nullptr;
    std::uint8_t* d_block_candidate_valid_ = nullptr;
    std::int32_t* d_block_frontier_vertices_ = nullptr;
    std::int64_t* d_block_frontier_scores_ = nullptr;
    std::uint8_t* d_block_selected_flags_ = nullptr;
    std::int32_t* d_block_trial_labels_ = nullptr;
    std::uint64_t* d_block_frontier_overflows_ = nullptr;
    std::uint64_t* d_block_edge_visits_ = nullptr;
    int* d_block_error_ = nullptr;
    void* d_select_temp_ = nullptr;
    void* d_sort_temp_ = nullptr;
    void* d_pair_sort_temp_ = nullptr;
    std::size_t select_temp_bytes_ = 0;
    std::size_t sort_temp_bytes_ = 0;
    std::size_t pair_sort_temp_bytes_ = 0;
    std::int64_t pair_raw_capacity_ = 0;
    int pair_bucket_count_ = 0;
    int pair_slot_count_ = 0;
    int block_pair_count_ = 0;
    int block_candidate_capacity_ = 0;
    bool initialized_ = false;
};
