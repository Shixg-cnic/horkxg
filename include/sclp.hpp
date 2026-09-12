#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <thrust/device_vector.h>

namespace sclp {

inline constexpr std::int64_t kBeta = 256;

struct WeightedGraph {
    std::vector<std::int64_t> offsets;
    std::vector<std::int32_t> neighbors;
    std::vector<std::uint64_t> edge_weights;
    std::vector<std::uint64_t> vertex_weights;

    std::int64_t vertices() const {
        return static_cast<std::int64_t>(vertex_weights.size());
    }
    std::int64_t edges() const {
        return static_cast<std::int64_t>(neighbors.size());
    }
};

struct DeviceWeightedGraph {
    thrust::device_vector<std::int64_t> offsets;
    thrust::device_vector<std::int32_t> neighbors;
    thrust::device_vector<std::uint64_t> edge_weights;
    thrust::device_vector<std::uint64_t> vertex_weights;

    std::int64_t vertices() const {
        return static_cast<std::int64_t>(vertex_weights.size());
    }
    std::int64_t edges() const {
        return static_cast<std::int64_t>(neighbors.size());
    }
};

struct DeviceAggregateResult {
    thrust::device_vector<std::int32_t> map;
    thrust::device_vector<std::uint64_t> vertex_weights;
    std::int32_t coarse_vertices = 0;
    std::uint64_t maximum_weight = 0;
    std::uint64_t capacity = 0;
};

class SclpWorkspace {
public:
    SclpWorkspace();
    ~SclpWorkspace();
    SclpWorkspace(SclpWorkspace&&) noexcept;
    SclpWorkspace& operator=(SclpWorkspace&&) noexcept;
    SclpWorkspace(const SclpWorkspace&) = delete;
    SclpWorkspace& operator=(const SclpWorkspace&) = delete;

    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct SclpStats {
    std::uint64_t capacity = 0;
    std::uint64_t lp_accepted = 0;
    std::uint64_t capacity_rejected = 0;
    std::uint64_t two_hop_merged = 0;
    std::uint64_t singleton_count = 0;
    std::uint64_t proposal_count = 0;
    std::uint64_t positive_gain_vertices = 0;
    std::uint64_t role_blocked_vertices = 0;
    std::uint64_t role_blocked_gain = 0;
    std::uint64_t singleton_with_favorite = 0;
    std::uint64_t pairable_singletons = 0;
    int rounds = 0;
    double affinity_seconds = 0.0;
    double admission_seconds = 0.0;
    double two_hop_seconds = 0.0;
    double compact_seconds = 0.0;
    double aggregate_seconds = 0.0;
};

struct ContractionTimings {
    double total_seconds = 0.0;
    double compact_seconds = 0.0;
    double sort_seconds = 0.0;
    double reduce_seconds = 0.0;
    double csr_build_seconds = 0.0;
    double vertex_weight_seconds = 0.0;
};

DeviceWeightedGraph make_device_weighted(const WeightedGraph& graph);
WeightedGraph copy_device_weighted(const DeviceWeightedGraph& graph);

DeviceAggregateResult aggregate(
    const DeviceWeightedGraph& graph, SclpWorkspace& workspace,
    int parts, std::uint32_t seed,
    int level, SclpStats& stats, bool diagnostics);

DeviceWeightedGraph contract(
    const DeviceWeightedGraph& fine, const DeviceAggregateResult& aggregate,
    SclpWorkspace& workspace, ContractionTimings& timings);

}  // namespace sclp
