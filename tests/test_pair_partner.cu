// Include the implementation only in this standalone test translation unit:
// exercise internal kernels without adding a production testing API.
#include "../src/refine.cu"

#include <thrust/host_vector.h>

int main() {
    try {
        using Types = gpart::ActiveTypes;
        using V = Types::VertexT;
        using W = Types::WeightT;
        using O = Types::OffsetT;
        constexpr int n = 521; // Partial final subwarp/warp block.
        constexpr std::uint32_t salt = 0x27d4eb2dU;
        const int degrees[] = {0, 1, 3, 4, 8, 31, 32, 33, 64, 256, 257, 4097};
        std::vector<O> offsets{0};
        std::vector<V> neighbors, parts(n), expected(n, -1);
        std::vector<W> weights;
        std::vector<std::uint32_t> targets(n);
        std::vector<std::int64_t> gains(n), expected_gain(n);
        std::vector<std::uint64_t> expected_count(n);
        for (int v = 0; v < n; ++v) {
            parts[v] = v % 2;
            targets[v] = v % 11 == 0 ? gpart::kInvalidTarget : 2 + v % 3;
            gains[v] = v % 7 - 4;
            for (int e = 0; e < degrees[v % 12]; ++e) {
                neighbors.push_back((v + e * 17) % n);
                weights.push_back(1 + e % 5);
            }
            offsets.push_back(static_cast<O>(neighbors.size()));
        }
        const auto mix = [](std::uint32_t x) {
            x ^= x >> 16; x *= 0x7feb352dU;
            x ^= x >> 15; x *= 0x846ca68bU;
            return x ^ (x >> 16);
        };
        // Original serial rule: count every eligible edge, including repeated
        // neighbors; gain > tie > smaller neighbor ID is the exact ordering.
        for (int v = 0; v < n; ++v) {
            std::uint32_t best_tie = 0;
            if (targets[v] == gpart::kInvalidTarget) continue;
            for (auto e = offsets[v]; e < offsets[v + 1]; ++e) {
                const auto u = neighbors[e];
                if (u == v || parts[u] != parts[v] || targets[u] != targets[v] ||
                    (gains[v] > 0 && gains[u] > 0)) continue;
                const auto gain = gains[v] + gains[u] + 2 * std::int64_t(weights[e]);
                if (gain <= 0) continue;
                ++expected_count[v];
                const auto tie = mix(mix(std::min(v, u)) ^ mix(std::max(v, u)) ^
                                     mix(targets[v]) ^ salt);
                if (gain > expected_gain[v] ||
                    (gain == expected_gain[v] &&
                     (tie > best_tie || (tie == best_tie && u < expected[v])))) {
                    expected[v] = u; expected_gain[v] = gain; best_tie = tie;
                }
            }
        }
        thrust::device_vector<O> d_offsets = offsets;
        thrust::device_vector<V> d_neighbors = neighbors, d_parts = parts, partner(n);
        thrust::device_vector<W> d_weights = weights;
        thrust::device_vector<std::uint32_t> d_targets = targets;
        thrust::device_vector<std::int64_t> d_gains = gains, pair_gain(n);
        thrust::device_vector<std::uint64_t> count(n);
#define PAIR_ARGS n, thrust::raw_pointer_cast(d_offsets.data()), \
    thrust::raw_pointer_cast(d_neighbors.data()), thrust::raw_pointer_cast(d_weights.data()), \
    thrust::raw_pointer_cast(d_parts.data()), thrust::raw_pointer_cast(d_targets.data()), \
    thrust::raw_pointer_cast(d_gains.data()), salt, thrust::raw_pointer_cast(partner.data()), \
    thrust::raw_pointer_cast(pair_gain.data()), thrust::raw_pointer_cast(count.data())
        gpart::pair_partner_kernel<4><<<(n + 63) / 64, 256>>>(PAIR_ARGS);
        CUDA_CHECK(cudaGetLastError());
        gpart::pair_partner_kernel<32><<<(n + 7) / 8, 256>>>(PAIR_ARGS);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
#undef PAIR_ARGS
        const thrust::host_vector<V> actual = partner;
        const thrust::host_vector<std::int64_t> actual_gain = pair_gain;
        const thrust::host_vector<std::uint64_t> actual_count = count;
        for (int v = 0; v < n; ++v) {
            if (actual[v] != expected[v] || actual_gain[v] != expected_gain[v] ||
                actual_count[v] != expected_count[v]) {
                throw std::runtime_error("partner serial equivalence failed at " +
                                         std::to_string(v));
            }
        }
        // Validation uses the old exact limit, including its fallback for Big.
        gpart::WeightedGraph<Types> graph;
        graph.edge_weights = {W{1}, W{2}};
        gpart::validate_pair_gain_range(graph);
        if constexpr (sizeof(W) == 8) {
            constexpr auto limit = std::uint64_t(INT64_MAX) / 2;
            graph.edge_weights = {static_cast<W>(limit)};
            gpart::validate_pair_gain_range(graph);
            for (const auto& invalid : std::vector<std::vector<W>>{
                     {static_cast<W>(limit + 1)},
                     {static_cast<W>(limit), W{1}},
                     {std::numeric_limits<W>::max(), W{1}}}) {
                graph.edge_weights = invalid;
                bool rejected = false;
                try { gpart::validate_pair_gain_range(graph); }
                catch (const std::overflow_error&) { rejected = true; }
                if (!rejected) throw std::runtime_error("unsafe pair gain accepted");
            }
        }
        std::cout << "pair_partner_serial_equivalence_ok=1 graph_type="
                  << gpart::kActiveGraphName << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
