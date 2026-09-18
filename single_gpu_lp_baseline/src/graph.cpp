#include "graph.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

std::int64_t element_count(const std::string& path, std::size_t element_size) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open " + path);
    const auto bytes = input.tellg();
    if (bytes < 0 || bytes % static_cast<std::streamoff>(element_size) != 0) {
        throw std::runtime_error("invalid binary size " + path);
    }
    return static_cast<std::int64_t>(
        bytes / static_cast<std::streamoff>(element_size));
}

template <typename T>
std::vector<T> read_all(const std::string& path, std::int64_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path);
    std::vector<T> values(static_cast<std::size_t>(count));
    if (count > 0) {
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(count * sizeof(T)));
    }
    if (!input) throw std::runtime_error("cannot read " + path);
    return values;
}

}

void CSRGraph::load(const std::string& indptr_path,
                    const std::string& indices_path) {
    const auto offset_count = element_count(indptr_path, sizeof(std::int64_t));
    edges_ = element_count(indices_path, sizeof(std::int64_t));
    if (offset_count < 1) throw std::runtime_error("empty CSR indptr");
    vertices_ = offset_count - 1;
    offsets_ = read_all<Edge>(indptr_path, offset_count);
    if (offsets_.front() != 0 || offsets_.back() != edges_) {
        throw std::runtime_error("CSR offsets do not match indices");
    }

    std::ifstream input(indices_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + indices_path);
    neighbors_.resize(static_cast<std::size_t>(edges_));
    constexpr std::size_t chunk = 1U << 20;
    std::vector<std::int64_t> raw(chunk);
    std::int64_t done = 0;
    while (done < edges_) {
        const auto take = static_cast<std::size_t>(
            std::min<std::int64_t>(chunk, edges_ - done));
        input.read(reinterpret_cast<char*>(raw.data()),
                   static_cast<std::streamsize>(take * sizeof(std::int64_t)));
        if (!input) throw std::runtime_error("cannot read " + indices_path);
        for (std::size_t i = 0; i < take; ++i) {
            neighbors_[static_cast<std::size_t>(done) + i] =
                static_cast<std::int32_t>(raw[i]);
        }
        done += static_cast<std::int64_t>(take);
    }
    std::cout << "vertices=" << vertices_ << " edge_entries=" << edges_
              << " device_count=1\n";
}
