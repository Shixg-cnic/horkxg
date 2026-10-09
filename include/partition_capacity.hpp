#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace gpart {

// Integer loads must not exceed alpha * total / k. Rounding up silently
// relaxes that constraint on the small graphs used in correctness tests.
inline std::uint64_t partition_capacity(
    std::uint64_t total, int parts, double imbalance_ratio) {
    if (parts < 2 || parts > 32 || !std::isfinite(imbalance_ratio) || imbalance_ratio < 1.0)
        throw std::invalid_argument("invalid partition capacity options");
    const long double raw = static_cast<long double>(total) * imbalance_ratio / parts;
    if (!std::isfinite(raw) || raw >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
        throw std::overflow_error("partition capacity exceeds uint64");
    const auto capacity = static_cast<std::uint64_t>(std::floor(raw));
    if (capacity < total / static_cast<unsigned>(parts) + (total % parts != 0))
        throw std::invalid_argument("integer partition capacity cannot hold total weight");
    return capacity;
}

}  // namespace gpart
