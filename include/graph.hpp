#pragma once

#include <cstdint>
#include <string>
#include <vector>

class CSRGraph {
public:
    using Vertex = std::int64_t;
    using Edge = std::int64_t;
    using Neighbor = std::int32_t;

    void load(const std::string& indptr_path, const std::string& indices_path);

    Vertex vertices() const { return vertices_; }
    Edge edges() const { return edges_; }
    const std::vector<Edge>& offsets() const { return offsets_; }
    const std::vector<Neighbor>& neighbors() const { return neighbors_; }

private:
    Vertex vertices_ = 0;
    Edge edges_ = 0;
    std::vector<Edge> offsets_;
    std::vector<Neighbor> neighbors_;
};
