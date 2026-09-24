#pragma once

#include <cstdint>
#include <limits>

#ifdef __CUDACC__
#define GPART_RECORD_HD __host__ __device__
#else
#define GPART_RECORD_HD
#endif

namespace gpart::detail {

// Lossless scratch representation, not a change to affinities or sort order.
// Reserve an unused vertex ID so UINT64_MAX remains a distinct invalid record.
struct SclpRecordLayout {
    int vertex_bits = 32;
    int weight_bits = 0;
    std::uint64_t weight_mask = 0;

    static SclpRecordLayout checked(
        std::int64_t vertices, std::uint64_t degree_per_weight,
        std::uint64_t maximum_vertex_weight) {
        if (vertices <= 0 || vertices > std::numeric_limits<std::int32_t>::max() ||
            degree_per_weight == std::numeric_limits<std::uint64_t>::max())
            return {};
        int bits = 0;
        for (auto value = vertices; value; value >>= 1) ++bits;
        const int weight_bits = 64 - 2 * bits;
        const auto mask = (std::uint64_t{1} << weight_bits) - 1;
        if (maximum_vertex_weight && degree_per_weight > mask / maximum_vertex_weight)
            return {};
        return {bits, weight_bits, mask};
    }

    GPART_RECORD_HD bool packed() const { return weight_bits != 0; }
    GPART_RECORD_HD std::uint64_t encode(
        std::uint32_t source, std::uint32_t target, std::uint64_t weight) const {
        if (!packed()) return (std::uint64_t{source} << 32) | target;
        return (((std::uint64_t{source} << vertex_bits) | target) << weight_bits) | weight;
    }
    GPART_RECORD_HD std::uint32_t source(std::uint64_t record) const {
        return static_cast<std::uint32_t>(record >> (packed() ? weight_bits + vertex_bits : 32));
    }
    GPART_RECORD_HD std::uint32_t target(std::uint64_t record) const {
        return packed() ? static_cast<std::uint32_t>((record >> weight_bits) &
            ((std::uint64_t{1} << vertex_bits) - 1)) : static_cast<std::uint32_t>(record);
    }
};

struct SclpRecordKey {
    int weight_bits;
    GPART_RECORD_HD std::uint64_t operator()(std::uint64_t record) const {
        return record >> weight_bits;
    }
};

// reduce_by_key invokes this only within one (source,target) group. A checked
// row-weight bound proves that the sum cannot carry into the encoded key.
struct SclpRecordSum {
    std::uint64_t weight_mask;
    GPART_RECORD_HD std::uint64_t operator()(std::uint64_t a, std::uint64_t b) const {
        return (a & ~weight_mask) | ((a & weight_mask) + (b & weight_mask));
    }
};

} // namespace gpart::detail
#undef GPART_RECORD_HD
