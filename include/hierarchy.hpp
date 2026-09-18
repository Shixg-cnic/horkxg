#pragma once

#include "graph.hpp"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpart {

template <typename Types>
struct Hierarchy {
    using VertexT = typename Types::VertexT;

    std::vector<WeightedGraph<Types>> levels;
    std::vector<std::vector<VertexT>> fine_to_coarse;
    std::string stop_reason = "capacity_floor";
    double hierarchy_loop_seconds = 0.0;
    double snapshot_seconds = 0.0;

    std::size_t layer_count() const { return levels.size(); }
};

// Production ownership: retain each level and its projection map on device.
// The host hierarchy remains available for export and compatibility tests.
template <typename Types>
struct DeviceHierarchy {
    std::vector<DeviceWeightedGraph<Types>> levels;
    std::vector<thrust::device_vector<typename Types::VertexT>> fine_to_coarse;
    std::string stop_reason = "capacity_floor";
    double hierarchy_loop_seconds = 0.0;
    double snapshot_seconds = 0.0;
    std::size_t layer_count() const { return levels.size(); }
};

// Preserve the existing Jet-compatible int32 on-disk representation. Runtime
// graph widths are intentionally independent of the hierarchy file format.
template <typename Types>
void write_jet_hierarchy(
    const Hierarchy<Types>& hierarchy, const std::string& path) {
    const auto& levels = hierarchy.levels;
    const auto& maps = hierarchy.fine_to_coarse;
    if (levels.empty() || maps.size() + 1 != levels.size()) {
        throw std::runtime_error("cannot write an incomplete Jet hierarchy");
    }
    const auto temporary = path + ".tmp";
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open hierarchy temporary file");
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
                const auto wide = static_cast<std::int64_t>(offset);
                if (wide < 0 || wide > std::numeric_limits<std::int32_t>::max()) {
                    throw std::runtime_error("Jet row offset exceeds int32");
                }
                const auto value = static_cast<std::int32_t>(wide);
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            static_assert(
                sizeof(typename Types::VertexT) == sizeof(std::int32_t),
                "Jet hierarchy neighbor ids require 32-bit storage");
            output.write(
                reinterpret_cast<const char*>(graph.neighbors.data()),
                static_cast<std::streamsize>(
                    static_cast<std::int64_t>(m) * sizeof(std::int32_t)));
            for (const auto weight : graph.edge_weights) {
                const auto wide = static_cast<std::uint64_t>(weight);
                if (wide > static_cast<std::uint64_t>(
                               std::numeric_limits<std::int32_t>::max())) {
                    throw std::runtime_error("Jet edge weight exceeds int32");
                }
                const auto value = static_cast<std::int32_t>(wide);
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            for (const auto weight : graph.vertex_weights) {
                const auto wide = static_cast<std::uint64_t>(weight);
                if (wide > static_cast<std::uint64_t>(
                               std::numeric_limits<std::int32_t>::max())) {
                    throw std::runtime_error("Jet vertex weight exceeds int32");
                }
                const auto value = static_cast<std::int32_t>(wide);
                output.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
            if (i > 0) {
                const auto& map = maps[i - 1];
                if (map.size() !=
                    static_cast<std::size_t>(levels[i - 1].vertices())) {
                    throw std::runtime_error("Jet map has the wrong fine-level length");
                }
                for (const auto coarse : map) {
                    const auto value = static_cast<std::int32_t>(coarse);
                    if (value < 0 || value >= n) {
                        throw std::runtime_error(
                            "Jet map contains an invalid coarse id");
                    }
                    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
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

}  // namespace gpart
