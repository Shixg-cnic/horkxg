#pragma once

#include "hierarchy.hpp"
#include "refine.hpp"

#include <vector>

namespace gpart {

template <typename Types>
std::vector<typename Types::VertexT> uncoarsen(
    const Hierarchy<Types>& hierarchy,
    const std::vector<typename Types::VertexT>& coarsest_partition,
    const RefineOptions& options);

}  // namespace gpart
