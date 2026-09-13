#pragma once

#include "graph.hpp"
#include "hierarchy.hpp"

#include <cstdint>
#include <memory>

#include <thrust/device_vector.h>

namespace gpart {

inline constexpr std::int64_t kBeta = 256;

struct CoarsenOptions {
    int parts = 4;
    std::uint32_t seed = 0;
    double stop_contraction_ratio = 0.90;
    int max_levels = 24;
    bool strict_verify = false;
};

template <typename Types>
struct DeviceAggregateResult {
    thrust::device_vector<typename Types::VertexT> map;
    thrust::device_vector<typename Types::WeightT> vertex_weights;
    typename Types::VertexT coarse_vertices = 0;
    std::uint64_t maximum_weight = 0;
    std::uint64_t capacity = 0;
};

template <typename Types>
class SclpWorkspace {
public:
    SclpWorkspace();
    ~SclpWorkspace();
    SclpWorkspace(SclpWorkspace&&) noexcept;
    SclpWorkspace& operator=(SclpWorkspace&&) noexcept;
    SclpWorkspace(const SclpWorkspace&) = delete;
    SclpWorkspace& operator=(const SclpWorkspace&) = delete;

    void reserve_contract_buffers(std::int64_t max_edges);

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

template <typename Types>
DeviceAggregateResult<Types> aggregate(
    const DeviceWeightedGraph<Types>& graph, SclpWorkspace<Types>& workspace,
    int parts, std::uint32_t seed,
    int level, SclpStats& stats, bool diagnostics);

template <typename Types>
DeviceWeightedGraph<Types> contract(
    DeviceWeightedGraph<Types> fine,
    DeviceAggregateResult<Types>&& aggregate,
    SclpWorkspace<Types>& workspace, ContractionTimings& timings);

template <typename Types>
Hierarchy<Types> coarsen(
    const WeightedGraph<Types>& graph, const CoarsenOptions& options);

}  // namespace gpart
