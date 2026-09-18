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

// Consumes device levels/maps and labels; the result keeps the final partition
// on device so output download can be timed separately from partitioning.
template <typename Types>
DeviceUncoarsenResult<Types> uncoarsen(
    DeviceHierarchy<Types>& hierarchy,
    thrust::device_vector<typename Types::VertexT>&& coarsest_partition,
    const RefineOptions& options);

}  // namespace gpart
