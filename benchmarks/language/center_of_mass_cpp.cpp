#include "center_of_mass_cpp.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace magtrack {
namespace benchmarks {

namespace {

inline bool is_background_mode(const std::string &requested,
                               const std::string &expected) {
    return requested == expected;
}

inline double compute_mean(const double *data, std::size_t count) {
    double sum = std::accumulate(data, data + count, 0.0);
    return sum / static_cast<double>(count);
}

inline double compute_median(const double *data, std::size_t count) {
    std::vector<double> buffer(data, data + count);
    auto mid_upper = buffer.begin() + static_cast<std::ptrdiff_t>(count / 2);
    std::nth_element(buffer.begin(), mid_upper, buffer.end());
    if (count % 2 != 0) {
        return *mid_upper;
    }
    auto mid_lower = buffer.begin() + static_cast<std::ptrdiff_t>(count / 2 - 1);
    std::nth_element(buffer.begin(), mid_lower, mid_upper + 1);
    return (*mid_lower + *mid_upper) * 0.5;
}

}  // namespace

void center_of_mass(
    const double *stack,
    std::size_t width,
    std::size_t height,
    std::size_t n_images,
    const std::string &background,
    double *x_out,
    double *y_out) {
    if (stack == nullptr || width == 0 || height == 0 || n_images == 0) {
        return;
    }

    const bool background_none = background == "none" || background.empty();
    const bool background_mean = is_background_mode(background, "mean");
    const bool background_median = is_background_mode(background, "median");

    if (!(background_none || background_mean || background_median)) {
        throw std::invalid_argument("Unsupported background mode: " + background);
    }

    const std::size_t image_size = width * height;
    std::vector<double> column_sums(width, 0.0);
    std::vector<double> adjusted_image(image_size, 0.0);

    const double nan_value = std::numeric_limits<double>::quiet_NaN();

    for (std::size_t image_idx = 0; image_idx < n_images; ++image_idx) {
        const double *image = stack + image_idx * image_size;
        const double *active_image = image;

        double background_value = 0.0;
        if (background_mean) {
            background_value = compute_mean(image, image_size);
        } else if (background_median) {
            background_value = compute_median(image, image_size);
        }

        if (!background_none) {
            std::transform(
                image,
                image + image_size,
                adjusted_image.begin(),
                [background_value](double value) {
                    return std::fabs(value - background_value);
                });
            active_image = adjusted_image.data();
        }

        std::fill(column_sums.begin(), column_sums.end(), 0.0);

        double total_mass = 0.0;
        double y_weighted_sum = 0.0;

        for (std::size_t r = 0; r < height; ++r) {
            const std::size_t row_offset = r * width;
            const double *row_start = active_image + row_offset;
            double row_sum = std::accumulate(row_start, row_start + width, 0.0);
            total_mass += row_sum;
            y_weighted_sum += static_cast<double>(r) * row_sum;

            for (std::size_t c = 0; c < width; ++c) {
                column_sums[c] += row_start[c];
            }
        }

        double x_weighted_sum = 0.0;
        for (std::size_t c = 0; c < width; ++c) {
            x_weighted_sum += static_cast<double>(c) * column_sums[c];
        }

        if (total_mass == 0.0) {
            if (x_out != nullptr) {
                x_out[image_idx] = nan_value;
            }
            if (y_out != nullptr) {
                y_out[image_idx] = nan_value;
            }
        } else {
            if (x_out != nullptr) {
                x_out[image_idx] = x_weighted_sum / total_mass;
            }
            if (y_out != nullptr) {
                y_out[image_idx] = y_weighted_sum / total_mass;
            }
        }
    }
}

std::vector<double> center_of_mass_x(
    const double *stack,
    std::size_t width,
    std::size_t height,
    std::size_t n_images,
    const std::string &background) {
    std::vector<double> x(n_images, std::numeric_limits<double>::quiet_NaN());
    center_of_mass(stack, width, height, n_images, background, x.data(), nullptr);
    return x;
}

std::vector<double> center_of_mass_y(
    const double *stack,
    std::size_t width,
    std::size_t height,
    std::size_t n_images,
    const std::string &background) {
    std::vector<double> y(n_images, std::numeric_limits<double>::quiet_NaN());
    center_of_mass(stack, width, height, n_images, background, nullptr, y.data());
    return y;
}

}  // namespace benchmarks
}  // namespace magtrack
