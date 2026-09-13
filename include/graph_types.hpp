#pragma once

#include <cstdint>

namespace gpart {

struct StandardTypes {
    using VertexT = std::int32_t;
    using OffsetT = std::int32_t;
    using WeightT = std::uint32_t;
};

struct BigTypes {
    using VertexT = std::int32_t;
    using OffsetT = std::int64_t;
    using WeightT = std::uint64_t;
};

#if defined(GPART_STANDARD_GRAPH) && defined(GPART_BIG_GRAPH)
#error "Define exactly one of GPART_STANDARD_GRAPH or GPART_BIG_GRAPH"
#elif defined(GPART_STANDARD_GRAPH)
using ActiveTypes = StandardTypes;
inline constexpr const char* kActiveGraphName = "standard";
#elif defined(GPART_BIG_GRAPH)
using ActiveTypes = BigTypes;
inline constexpr const char* kActiveGraphName = "big";
#endif

}  // namespace gpart
