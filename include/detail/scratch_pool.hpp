#pragma once

#include "check.hpp"
#include <cuda_runtime_api.h>
#include <thrust/device_allocator.h>
#include <thrust/device_vector.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace gpart::detail {

struct ScratchPoolState {
    cudaMemPool_t pool = nullptr;
    ScratchPoolState() {
        cudaMemPoolProps properties{};
        properties.allocType = cudaMemAllocationTypePinned;
        properties.location.type = cudaMemLocationTypeDevice;
        CUDA_CHECK(cudaGetDevice(&properties.location.id));
        CUDA_CHECK(cudaMemPoolCreate(&pool, &properties));
        std::uint64_t retention = UINT64_MAX;
        const auto status = cudaMemPoolSetAttribute(
            pool, cudaMemPoolAttrReleaseThreshold, &retention);
        if (status != cudaSuccess) {
            cudaMemPoolDestroy(pool);
            pool = nullptr;
            CUDA_CHECK(status);
        }
    }
    ~ScratchPoolState() { if (pool) cudaMemPoolDestroy(pool); }
};

inline thread_local std::shared_ptr<ScratchPoolState> active_scratch_pool;

// Explicit pipeline lifetime; never preallocate or warm memory before timing.
// Allocators keep the pool alive if a scratch vector outlives the session.
class ScratchPoolSession {
    std::shared_ptr<ScratchPoolState> previous_ = active_scratch_pool;
    std::shared_ptr<ScratchPoolState> state_ = std::make_shared<ScratchPoolState>();
public:
    ScratchPoolSession() { active_scratch_pool = state_; }
    ScratchPoolSession(const ScratchPoolSession&) = delete;
    ScratchPoolSession& operator=(const ScratchPoolSession&) = delete;
    ~ScratchPoolSession() { if (state_) active_scratch_pool = previous_; }
    void finish() {
        if (!state_) return;
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemPoolTrimTo(state_->pool, 0));
        active_scratch_pool = previous_;
        state_.reset();
    }
};

template <typename T>
class ScratchAllocator : public thrust::device_allocator<T> {
    template <typename> friend class ScratchAllocator;
    std::shared_ptr<ScratchPoolState> state_ = active_scratch_pool;
public:
    using Base = thrust::device_allocator<T>;
    using pointer = typename Base::pointer;
    using is_always_equal = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    template <typename U> struct rebind { using other = ScratchAllocator<U>; };
    __host__ ScratchAllocator() {}
    __host__ ScratchAllocator(const ScratchAllocator& other) : Base(other), state_(other.state_) {}
    __host__ ScratchAllocator(ScratchAllocator&& other) noexcept
        : Base(other), state_(std::move(other.state_)) {}
    __host__ ScratchAllocator& operator=(const ScratchAllocator& other) {
        state_ = other.state_; return *this;
    }
    __host__ ScratchAllocator& operator=(ScratchAllocator&& other) noexcept {
        state_ = std::move(other.state_); return *this;
    }
    __host__ ~ScratchAllocator() {}
    template <typename U>
    __host__ ScratchAllocator(const ScratchAllocator<U>& other) : state_(other.state_) {}
    pointer allocate(std::size_t count) {
        if (!state_) return Base::allocate(count);
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();
        void* allocation = nullptr;
        CUDA_CHECK(cudaMallocFromPoolAsync(&allocation, count * sizeof(T), state_->pool, nullptr));
        // Scratch kernels use the default stream; make allocation ready before
        // returning to callers, preserving the previous blocking interface.
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        return pointer(static_cast<T*>(allocation));
    }
    void deallocate(pointer allocation, std::size_t count) {
        if (!state_) { Base::deallocate(allocation, count); return; }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFreeAsync(thrust::raw_pointer_cast(allocation), nullptr));
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    template <typename U> bool operator==(const ScratchAllocator<U>& other) const {
        return state_ == other.state_;
    }
    template <typename U> bool operator!=(const ScratchAllocator<U>& other) const {
        return !(*this == other);
    }
};

template <typename T>
using ScratchVector = thrust::device_vector<T, ScratchAllocator<T>>;

} // namespace gpart::detail
