#include "partition_capacity.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>

void require(bool result, const char* message) {
    if (!result) throw std::runtime_error(message);
}

int main() {
    try {
        require(gpart::partition_capacity(6, 2, 1.10) == 3, "capacity rounded up");
        require(gpart::partition_capacity(100, 4, 1.03) == 25, "3% capacity is wrong");
        require(gpart::partition_capacity(1000, 4, 1.03) == 257, "fractional capacity is wrong");
        require(gpart::partition_capacity(6, 2, 1.34) == 4, "four-vertex fixture capacity");
        for (auto [total, parts, ratio] : {
            std::tuple<std::uint64_t, int, double>{4, 3, 1.03},
            {100, 0, 1.03}, {100, 33, 1.03}, {100, 4, 0.99},
            {100, 4, std::numeric_limits<double>::quiet_NaN()}}) {
            bool rejected = false;
            try { gpart::partition_capacity(total, parts, ratio); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid or infeasible capacity accepted");
        }
        bool overflow = false;
        try { gpart::partition_capacity(std::numeric_limits<std::uint64_t>::max(), 2, 4.0); }
        catch (const std::overflow_error&) { overflow = true; }
        require(overflow, "capacity overflow accepted");
        std::cout << "partition_capacity_ok=1\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
