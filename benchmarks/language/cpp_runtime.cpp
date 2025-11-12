// Build with:
//   g++ -O3 -march=native -std=c++17 -DNDEBUG -ffast-math \
//       benchmarks/language/cpp_runtime.cpp -o cpp_runtime

#include "center_of_mass_cpp.h"

#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
    const std::size_t width = 100;
    const std::size_t height = 100;
    const std::size_t n_images = 100;
    const std::size_t total_values = width * height * n_images;

    std::vector<double> stack(total_values, 1.0);
    std::vector<double> x(n_images, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> y(n_images, std::numeric_limits<double>::quiet_NaN());

    try {
        for (std::size_t i = 0; i < 100; ++i) {
            magtrack::benchmarks::center_of_mass(
                stack.data(), width, height, n_images, "none", x.data(), y.data());
        }

        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < 10000; ++i) {
            magtrack::benchmarks::center_of_mass(
                stack.data(), width, height, n_images, "none", x.data(), y.data());
        }
        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> elapsed = end - start;
        std::cout << "Runtime: " << elapsed.count() << " seconds" << std::endl;
    } catch (const std::exception &err) {
        std::cerr << "Error: " << err.what() << std::endl;
        return 1;
    }

    return 0;
}
