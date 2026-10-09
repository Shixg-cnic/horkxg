#include "initial_partition.hpp"
#include "check.hpp"
#include "partition_capacity.hpp"

#include <thrust/binary_search.h>
#include <thrust/extrema.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gpart {
namespace {
using U64 = unsigned long long;
template <typename T> T* ptr(thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}
template <typename T> const T* ptr(const thrust::device_vector<T>& v) {
    return thrust::raw_pointer_cast(v.data());
}

struct CheckedSum {
    __host__ __device__ U64 operator()(U64 a, U64 b) const {
        return ~U64{0} - a < b ? ~U64{0} : a + b;
    }
};
__device__ unsigned hash(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}

struct Moments { double mass, sum, squares; };
struct Normalization { double mean, scale; };

__device__ Moments warp_sum(Moments m) {
    for (int s = 16; s; s /= 2) {
        m.mass += __shfl_down_sync(0xffffffffU, m.mass, s);
        m.sum += __shfl_down_sync(0xffffffffU, m.sum, s);
        m.squares += __shfl_down_sync(0xffffffffU, m.squares, s);
    }
    return m;
}

// All 256 threads participate, including the padding in the last block.
// A fixed reduction tree makes repeated runs independent of scheduling.
__device__ Moments block_sum(Moments m, Moments* shared) {
    int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
    m = warp_sum(m);
    if (!lane) shared[warp] = m;
    __syncthreads();
    m = threadIdx.x < 8 ? shared[lane] : Moments{0, 0, 0};
    if (!warp) m = warp_sum(m);
    return m;
}

__global__ void finish_moments(int count, const Moments* partials, Normalization* norm) {
    __shared__ Moments shared[8];
    Moments m{0, 0, 0};
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        m.mass += partials[i].mass;
        m.sum += partials[i].sum;
        m.squares += partials[i].squares;
    }
    m = block_sum(m, shared);
    if (!threadIdx.x) {
        double mean = m.sum / m.mass;
        double variance = fmax(0.0, m.squares / m.mass - mean * mean);
        *norm = {mean, variance > 1e-30 ? 1.0 / sqrt(variance) : 1.0};
    }
}

template <typename O, typename W>
__global__ void spectral_start(int count, const int* order, const int* labels,
    int group, const O* offsets, const int* neighbors, const W* weights,
    double* degree, double* x, unsigned seed, Moments* partials) {
    __shared__ Moments shared[8];
    int lane = threadIdx.x & 31;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    double d = 0;
    if (i < count) {
        int v = order[i];
        for (std::int64_t e = static_cast<std::int64_t>(offsets[v]) + lane;
             e < offsets[v + 1]; e += 32)
            if (labels[neighbors[e]] == group) d += static_cast<double>(weights[e]);
    }
    for (int s = 16; s; s /= 2) d += __shfl_down_sync(0xffffffffU, d, s);
    Moments m{0, 0, 0};
    if (!lane && i < count) {
        int v = order[i];
        double a = static_cast<double>(hash(v ^ seed)) / 4294967296.0 - 0.5;
        degree[v] = d;
        x[v] = a;
        double mass = d > 0 ? d : 1.0;
        m = {mass, mass * a, mass * a * a};
    }
    m = block_sum(m, shared);
    if (!threadIdx.x) partials[blockIdx.x] = m;
}

template <typename O, typename W>
__global__ void spectral_step(int count, const int* order, const int* labels,
    int group, const O* offsets, const int* neighbors, const W* weights,
    const double* degree, const double* x, double* y, Moments* partials) {
    __shared__ Moments shared[8];
    int lane = threadIdx.x & 31;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    double sum = 0;
    if (i < count) {
        int v = order[i];
        for (std::int64_t e = static_cast<std::int64_t>(offsets[v]) + lane;
             e < offsets[v + 1]; e += 32) {
            int u = neighbors[e];
            if (labels[u] == group) sum += static_cast<double>(weights[e]) * x[u];
        }
    }
    for (int s = 16; s; s /= 2) sum += __shfl_down_sync(0xffffffffU, sum, s);
    Moments m{0, 0, 0};
    if (!lane && i < count) {
        int v = order[i];
        double d = degree[v], a = d > 0 ? 0.5 * (x[v] + sum / d) : x[v];
        y[v] = a;
        double mass = d > 0 ? d : 1.0;
        m = {mass, mass * a, mass * a * a};
    }
    m = block_sum(m, shared);
    if (!threadIdx.x) partials[blockIdx.x] = m;
}

__global__ void normalize(int count, const int* order, double* x,
    const Normalization* norm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { int v = order[i]; x[v] = (x[v] - norm->mean) * norm->scale; }
}
struct SpectralOrder {
    const double* x;
    __device__ bool operator()(int a, int b) const {
        return x[a] < x[b] || (x[a] == x[b] && a < b);
    }
};
__global__ void assign_split(int count, const int* order, int boundary,
    int left, int right, int* labels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) labels[order[i]] = i < boundary ? left : right;
}
template <typename O, typename W>
__global__ void cut_terms(int n, const O* offsets, const int* neighbors,
    const W* weights, const int* labels, U64* terms) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    U64 sum = 0;
    for (O e = offsets[v]; e < offsets[v + 1]; ++e)
        if (labels[v] != labels[neighbors[e]]) sum += weights[e];
    terms[v] = sum;
}
__global__ void part_weights(int n, const int* labels, const U64* weights, U64* loads) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) atomicAdd(loads + labels[v], weights[v]);
}
} // namespace

template <typename Types>
thrust::device_vector<typename Types::VertexT> initial_partition_gpu(
    const DeviceWeightedGraph<Types>& graph, int parts, double imbalance,
    std::uint32_t seed) {
    const auto start = std::chrono::steady_clock::now();
    const auto n64 = graph.vertices();
    if (n64 <= 0 || n64 > std::numeric_limits<int>::max() - 255 || parts < 2 ||
        parts > n64 || parts > 32 ||
        !std::isfinite(imbalance) || imbalance < 1.0)
        throw std::invalid_argument("invalid GPU initial partition input");
    int n = static_cast<int>(n64);
    const U64 total = thrust::reduce(graph.vertex_weights.begin(),
        graph.vertex_weights.end(), U64{0}, CheckedSum{});
    const U64 edge_total = thrust::reduce(graph.edge_weights.begin(),
        graph.edge_weights.end(), U64{0}, CheckedSum{});
    if (!total || total == ~U64{0} || edge_total == ~U64{0})
        throw std::overflow_error("GPU initializer weight accumulation overflow");
    const U64 capacity = partition_capacity(total, parts, imbalance);
    auto max_weight = *thrust::max_element(graph.vertex_weights.begin(), graph.vertex_weights.end());
    if (static_cast<U64>(max_weight) > capacity)
        throw std::invalid_argument("a vertex exceeds partition capacity");

    thrust::device_vector<int> labels(n, 0), order(n);
    thrust::sequence(order.begin(), order.end());
    thrust::device_vector<double> degree(n), x(n), y(n);
    thrust::device_vector<Moments> partials((n + 7) / 8);
    thrust::device_vector<Normalization> norm(1);
    thrust::device_vector<U64> ordered_weights(n), prefix(n), vertex_weights(graph.vertex_weights);
    struct Task { int begin, count, first_part, parts; U64 weight; };
    std::vector<Task> tasks{{0, n, 0, parts, total}};
    while (!tasks.empty()) {
        Task task = tasks.back(); tasks.pop_back();
        if (task.parts == 1) {
            if (task.weight > capacity) throw std::runtime_error("GPU initializer infeasible balance");
            continue;
        }
        int count = task.count, blocks = (count + 255) / 256;
        auto begin = order.begin() + task.begin;
        int* ids = ptr(order) + task.begin;
        int spectral_blocks = (count + 7) / 8;
        spectral_start<<<spectral_blocks, 256>>>(count, ids, ptr(labels), task.first_part,
            ptr(graph.offsets), ptr(graph.neighbors), ptr(graph.edge_weights),
            ptr(degree), ptr(x), seed ^ static_cast<unsigned>(task.first_part), ptr(partials));
        CUDA_CHECK(cudaGetLastError());
        finish_moments<<<1, 256>>>(spectral_blocks, ptr(partials), ptr(norm));
        CUDA_CHECK(cudaGetLastError());
        // Fixed work, no parameter sweep. Remove the stationary component at
        // every iteration; lazy diffusion suppresses high-frequency modes.
        for (int iteration = 0; iteration <= 64; ++iteration) {
            normalize<<<blocks, 256>>>(count, ids, ptr(x), ptr(norm));
            CUDA_CHECK(cudaGetLastError());
            if (iteration == 64) break;
            spectral_step<<<spectral_blocks, 256>>>(count, ids, ptr(labels),
                task.first_part, ptr(graph.offsets), ptr(graph.neighbors),
                ptr(graph.edge_weights), ptr(degree), ptr(x), ptr(y), ptr(partials));
            CUDA_CHECK(cudaGetLastError());
            finish_moments<<<1, 256>>>(spectral_blocks, ptr(partials), ptr(norm));
            CUDA_CHECK(cudaGetLastError());
            x.swap(y);
        }
        thrust::sort(begin, begin + count, SpectralOrder{ptr(x)});
        thrust::gather(begin, begin + count, vertex_weights.begin(), ordered_weights.begin());
        thrust::inclusive_scan(ordered_weights.begin(), ordered_weights.begin() + count, prefix.begin());
        const int left_parts = task.parts / 2, right_parts = task.parts - left_parts;
        const U64 target = static_cast<U64>(static_cast<long double>(task.weight) * left_parts / task.parts);
        const U64 left_limit = capacity > task.weight / left_parts ? task.weight : capacity * left_parts;
        const U64 right_limit = capacity > task.weight / right_parts ? task.weight : capacity * right_parts;
        const U64 min_left = task.weight > right_limit ? task.weight - right_limit : 0;
        int boundary = static_cast<int>(thrust::lower_bound(prefix.begin(), prefix.begin() + count, target) - prefix.begin()) + 1;
        boundary = std::max(left_parts, std::min(count - right_parts, boundary));
        U64 left_weight = prefix[boundary - 1];
        if (left_weight > left_limit && boundary > left_parts) {
            --boundary; left_weight = prefix[boundary - 1];
        }
        if (left_weight < min_left && boundary < count - right_parts) {
            ++boundary; left_weight = prefix[boundary - 1];
        }
        if (left_weight > left_limit || left_weight < min_left)
            throw std::runtime_error("GPU spectral split cannot satisfy capacity");
        assign_split<<<blocks, 256>>>(count, ids, boundary, task.first_part,
            task.first_part + left_parts, ptr(labels));
        CUDA_CHECK(cudaGetLastError());
        tasks.push_back({task.begin + boundary, count - boundary,
            task.first_part + left_parts, right_parts, task.weight - left_weight});
        tasks.push_back({task.begin, boundary, task.first_part, left_parts, left_weight});
    }
    thrust::device_vector<U64> loads(parts, 0);
    part_weights<<<(n + 255) / 256, 256>>>(n, ptr(labels), ptr(vertex_weights), ptr(loads));
    CUDA_CHECK(cudaGetLastError());
    std::vector<U64> host_loads(parts);
    CUDA_CHECK(cudaMemcpy(host_loads.data(), ptr(loads), parts * sizeof(U64), cudaMemcpyDeviceToHost));
    if (*std::max_element(host_loads.begin(), host_loads.end()) > capacity)
        throw std::runtime_error("GPU initializer balance validation failed");
    cut_terms<<<(n + 255) / 256, 256>>>(n, ptr(graph.offsets), ptr(graph.neighbors),
        ptr(graph.edge_weights), ptr(labels), ptr(prefix));
    CUDA_CHECK(cudaGetLastError());
    U64 cut = thrust::reduce(prefix.begin(), prefix.end(), U64{0}) / 2;
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "initial_partition_backend=gpu_spectral_experimental\n"
        << "coarsest_vertices=" << n << '\n'
        << "coarsest_edge_entries=" << graph.edges() << '\n'
        << "initial_partition_parts=" << parts << '\n'
        << "initial_partition_edgecut=" << cut << '\n'
        << "initial_partition_seconds=" << std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count() << '\n';
    return labels;
}

template thrust::device_vector<ActiveTypes::VertexT> initial_partition_gpu<ActiveTypes>(
    const DeviceWeightedGraph<ActiveTypes>&, int, double, std::uint32_t);
} // namespace gpart
