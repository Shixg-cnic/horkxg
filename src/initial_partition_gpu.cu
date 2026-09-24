#include "initial_partition.hpp"
#include "check.hpp"

#include <thrust/binary_search.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

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

template <typename O, typename W>
__global__ void spectral_start(int count, const int* order, const int* labels,
    int group, const O* offsets, const int* neighbors, const W* weights,
    double* degree, double* x, unsigned seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int v = order[i];
    double d = 0;
    for (O e = offsets[v]; e < offsets[v + 1]; ++e)
        if (labels[neighbors[e]] == group) d += static_cast<double>(weights[e]);
    degree[v] = d > 0 ? d : 1.0;
    x[v] = static_cast<double>(hash(v ^ seed)) / 4294967296.0 - 0.5;
}

template <typename O, typename W>
__global__ void spectral_step(int count, const int* order, const int* labels,
    int group, const O* offsets, const int* neighbors, const W* weights,
    const double* x, double* y) {
    int lane = threadIdx.x & 31;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (i >= count) return;
    int v = order[i];
    double sum = 0, degree = 0;
    for (std::int64_t e = static_cast<std::int64_t>(offsets[v]) + lane;
         e < offsets[v + 1]; e += 32) {
        int u = neighbors[e];
        if (labels[u] == group) {
            double w = static_cast<double>(weights[e]);
            sum += w * x[u]; degree += w;
        }
    }
    for (int s = 16; s; s /= 2) {
        sum += __shfl_down_sync(0xffffffffU, sum, s);
        degree += __shfl_down_sync(0xffffffffU, degree, s);
    }
    if (!lane) y[v] = degree > 0 ? 0.5 * (x[v] + sum / degree) : x[v];
}

struct Moments { double mass, sum, squares; };
struct AddMoments {
    __host__ __device__ Moments operator()(Moments a, Moments b) const {
        return {a.mass + b.mass, a.sum + b.sum, a.squares + b.squares};
    }
};
struct GetMoments {
    const double* degree; const double* x;
    __device__ Moments operator()(int v) const {
        double d = degree[v], a = x[v]; return {d, d * a, d * a * a};
    }
};
__global__ void normalize(int count, const int* order, double* x,
    double mean, double scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { int v = order[i]; x[v] = (x[v] - mean) * scale; }
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
struct PartWeight {
    const int* labels; const U64* weights; int part;
    __device__ U64 operator()(int v) const {
        return labels[v] == part ? weights[v] : 0;
    }
};
} // namespace

template <typename Types>
thrust::device_vector<typename Types::VertexT> initial_partition_gpu(
    const DeviceWeightedGraph<Types>& graph, int parts, double imbalance,
    std::uint32_t seed) {
    const auto start = std::chrono::steady_clock::now();
    const auto n64 = graph.vertices();
    if (n64 <= 0 || n64 > std::numeric_limits<int>::max() - 255 || parts < 2 ||
        parts > n64 || parts > 32 || !std::isfinite(imbalance) || imbalance < 1.0)
        throw std::invalid_argument("invalid GPU initial partition input");
    int n = static_cast<int>(n64);
    const U64 total = thrust::reduce(graph.vertex_weights.begin(),
        graph.vertex_weights.end(), U64{0}, CheckedSum{});
    const U64 edge_total = thrust::reduce(graph.edge_weights.begin(),
        graph.edge_weights.end(), U64{0}, CheckedSum{});
    if (!total || total == ~U64{0} || edge_total == ~U64{0})
        throw std::overflow_error("GPU initializer weight accumulation overflow");
    long double raw_capacity = std::ceil(static_cast<long double>(total) * imbalance / parts);
    if (raw_capacity >= static_cast<long double>(~U64{0}))
        throw std::overflow_error("GPU initializer capacity overflow");
    U64 capacity = static_cast<U64>(raw_capacity);
    auto max_weight = *thrust::max_element(graph.vertex_weights.begin(), graph.vertex_weights.end());
    if (static_cast<U64>(max_weight) > capacity)
        throw std::invalid_argument("a vertex exceeds partition capacity");

    thrust::device_vector<int> labels(n, 0), order(n);
    thrust::sequence(order.begin(), order.end());
    thrust::device_vector<double> degree(n), x(n), y(n);
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
        spectral_start<<<blocks, 256>>>(count, ids, ptr(labels), task.first_part,
            ptr(graph.offsets), ptr(graph.neighbors), ptr(graph.edge_weights),
            ptr(degree), ptr(x), seed ^ static_cast<unsigned>(task.first_part));
        CUDA_CHECK(cudaGetLastError());
        // Fixed work, no parameter sweep. Remove the stationary component at
        // every iteration; lazy diffusion suppresses high-frequency modes.
        for (int iteration = 0; iteration <= 64; ++iteration) {
            Moments m = thrust::transform_reduce(begin, begin + count,
                GetMoments{ptr(degree), ptr(x)}, Moments{0, 0, 0}, AddMoments{});
            double mean = m.sum / m.mass;
            double variance = std::max(0.0, m.squares / m.mass - mean * mean);
            double scale = variance > 1e-30 ? 1.0 / std::sqrt(variance) : 1.0;
            normalize<<<blocks, 256>>>(count, ids, ptr(x), mean, scale);
            CUDA_CHECK(cudaGetLastError());
            if (iteration == 64) break;
            spectral_step<<<(count + 7) / 8, 256>>>(count, ids, ptr(labels),
                task.first_part, ptr(graph.offsets), ptr(graph.neighbors),
                ptr(graph.edge_weights), ptr(x), ptr(y));
            CUDA_CHECK(cudaGetLastError());
            x.swap(y);
        }
        thrust::sort(begin, begin + count, SpectralOrder{ptr(x)});
        thrust::gather(begin, begin + count, vertex_weights.begin(), ordered_weights.begin());
        thrust::inclusive_scan(ordered_weights.begin(), ordered_weights.begin() + count, prefix.begin());
        int left_parts = task.parts / 2, right_parts = task.parts - left_parts;
        U64 target = static_cast<U64>(static_cast<long double>(task.weight) * left_parts / task.parts);
        int boundary = static_cast<int>(thrust::lower_bound(prefix.begin(), prefix.begin() + count,
            target) - prefix.begin()) + 1;
        boundary = std::max(left_parts, std::min(count - right_parts, boundary));
        U64 left_weight = prefix[boundary - 1];
        // Choose the closest feasible prefix (whole vertices, never fractional
        // weights). Reject infeasible spectral ordering explicitly, no silent
        // over-capacity initialization or hidden CPU/METIS fallback.
        U64 left_limit = capacity > task.weight / left_parts ? task.weight : capacity * left_parts;
        U64 right_limit = capacity > task.weight / right_parts ? task.weight : capacity * right_parts;
        U64 min_left = task.weight > right_limit ? task.weight - right_limit : 0;
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
    for (int p = 0; p < parts; ++p) {
        U64 weight = thrust::transform_reduce(thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(n), PartWeight{ptr(labels), ptr(vertex_weights), p},
            U64{0}, thrust::plus<U64>());
        if (weight > capacity) throw std::runtime_error("GPU initializer balance validation failed");
    }
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
