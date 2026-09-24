#include "detail/sclp_record.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using gpart::detail::SclpRecordKey;
using gpart::detail::SclpRecordLayout;
using gpart::detail::SclpRecordSum;
using U64 = std::uint64_t;

void require(bool result, const char* message) {
    if (!result) throw std::runtime_error(message);
}

int main() {
    try {
        constexpr auto invalid = std::numeric_limits<U64>::max();
        for (std::int64_t n : {1LL, 2LL, 3LL, 4LL, 7LL, 8LL, 9LL,
                              1023LL, 1024LL, 4095LL, 4096LL,
                              2449029LL, 1295331LL, 2147483647LL}) {
            auto layout = SclpRecordLayout::checked(n, 1, 1);
            require(layout.packed(), "small weights should fit");
            SclpRecordKey key{layout.weight_bits};
            for (auto source : {std::int64_t{0}, n / 2, n - 1}) {
                for (auto target : {std::int64_t{0}, n / 2, n - 1}) {
                    for (U64 weight : {U64{0}, U64{1}, layout.weight_mask}) {
                        U64 record = layout.encode(source, target, weight);
                        require(layout.source(record) == source, "source roundtrip");
                        require(layout.target(record) == target, "target roundtrip");
                        require((record & layout.weight_mask) == weight, "weight roundtrip");
                        require(key(record) < key(invalid), "invalid sentinel collision");
                    }
                    SclpRecordSum sum{layout.weight_mask};
                    U64 record = sum(layout.encode(source, target, layout.weight_mask - 1),
                                     layout.encode(source, target, 1));
                    require(record == layout.encode(source, target, layout.weight_mask),
                            "sum carried into encoded key");
                }
            }
            require(SclpRecordLayout::checked(n, layout.weight_mask, 1).packed(), "exact bound rejected");
            require(!SclpRecordLayout::checked(n, layout.weight_mask + 1, 1).packed(), "overflow accepted");
            require(!SclpRecordLayout::checked(n, layout.weight_mask / 2 + 1, 2).packed(), "product overflow accepted");
            require(!SclpRecordLayout::checked(n, invalid, 0).packed(), "saturated density accepted");
        }
        require(!SclpRecordLayout::checked(0, 0, 1).packed(), "empty ID range");
        require(!SclpRecordLayout::checked(2147483648LL, 1, 1).packed(), "ID width overflow");

        // Compare stable (source,target) ordering and exact reduction to the
        // legacy key/value representation, including repeated/zero weights.
        std::mt19937 rng(0);
        auto layout = SclpRecordLayout::checked(2048, 1000000, 1);
        SclpRecordKey key{layout.weight_bits};
        SclpRecordSum sum{layout.weight_mask};
        std::vector<U64> records;
        std::vector<std::pair<U64, U64>> legacy;
        std::map<U64, U64> reference;
        for (int i = 0; i < 100000; ++i) {
            U64 source = rng() % 256, target = rng() % 256, weight = rng() % 257;
            U64 old_key = (source << 32) | target;
            records.push_back(layout.encode(source, target, weight));
            legacy.emplace_back(old_key, weight);
            reference[old_key] += weight;
        }
        std::stable_sort(records.begin(), records.end(), [key](U64 a, U64 b) { return key(a) < key(b); });
        std::stable_sort(legacy.begin(), legacy.end(), [](auto a, auto b) { return a.first < b.first; });
        for (std::size_t i = 0; i < records.size(); ++i) {
            U64 old_key = (U64{layout.source(records[i])} << 32) | layout.target(records[i]);
            require(old_key == legacy[i].first && (records[i] & layout.weight_mask) == legacy[i].second,
                    "stable sort semantics changed");
        }
        std::map<U64, U64> actual;
        for (std::size_t i = 0; i < records.size();) {
            U64 combined = records[i++];
            while (i < records.size() && key(records[i]) == key(combined))
                combined = sum(combined, records[i++]);
            U64 old_key = (U64{layout.source(combined)} << 32) | layout.target(combined);
            actual[old_key] = combined & layout.weight_mask;
        }
        require(actual == reference, "weighted reduction changed");
        std::cout << "sclp_record_host_ok=1\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
