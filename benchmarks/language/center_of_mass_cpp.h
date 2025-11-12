#ifndef BENCHMARKS_LANGUAGE_CENTER_OF_MASS_CPP_H
#define BENCHMARKS_LANGUAGE_CENTER_OF_MASS_CPP_H

#include <cstddef>
#include <string>
#include <vector>

namespace magtrack {
namespace benchmarks {

void center_of_mass(
    const double *stack,
    std::size_t width,
    std::size_t height,
    std::size_t n_images,
    const std::string &background,
    double *x_out,
    double *y_out);

std::vector<double> center_of_mass_x(
    const double *stack,
    std::size_t width,
    std::size_t height,
    std::size_t n_images,
    const std::string &background);

std::vector<double> center_of_mass_y(
    const double *stack,
    std::size_t width,
    std::size_t height,
    std::size_t n_images,
    const std::string &background);

}  // namespace benchmarks
}  // namespace magtrack

#endif  // BENCHMARKS_LANGUAGE_CENTER_OF_MASS_CPP_H
