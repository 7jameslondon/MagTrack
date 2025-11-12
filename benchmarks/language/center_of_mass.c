#include "center_of_mass.h"

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static int compare_doubles(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    if (da < db) {
        return -1;
    }
    if (da > db) {
        return 1;
    }
    return 0;
}

static bool strings_equal(const char *a, const char *b) {
    if (a == NULL || b == NULL) {
        return false;
    }
    return strcmp(a, b) == 0;
}

void center_of_mass(
    const double *stack,
    size_t width,
    size_t height,
    size_t n_images,
    const char *background,
    double *x_out,
    double *y_out) {
    if (width == 0 || height == 0 || n_images == 0) {
        return;
    }

    size_t image_size = width * height;
    bool background_none = strings_equal(background, "none") || background == NULL;
    bool background_mean = strings_equal(background, "mean");
    bool background_median = strings_equal(background, "median");

    if (!(background_none || background_mean || background_median)) {
        if (x_out != NULL) {
            for (size_t i = 0; i < n_images; ++i) {
                x_out[i] = NAN;
            }
        }
        if (y_out != NULL) {
            for (size_t i = 0; i < n_images; ++i) {
                y_out[i] = NAN;
            }
        }
        return;
    }

    double *column_sums = (double *)malloc(width * sizeof(double));
    if (column_sums == NULL) {
        return;
    }

    double *median_buffer = NULL;
    if (background_median) {
        median_buffer = (double *)malloc(image_size * sizeof(double));
        if (median_buffer == NULL) {
            free(column_sums);
            return;
        }
    }

    for (size_t image_idx = 0; image_idx < n_images; ++image_idx) {
        const double *image = stack + image_idx * image_size;
        double total_mass = 0.0;
        double x_num = 0.0;
        double y_num = 0.0;

        for (size_t c = 0; c < width; ++c) {
            column_sums[c] = 0.0;
        }

        double background_value = 0.0;
        if (background_mean) {
            for (size_t i = 0; i < image_size; ++i) {
                background_value += image[i];
            }
            background_value /= (double)image_size;
        } else if (background_median) {
            memcpy(median_buffer, image, image_size * sizeof(double));
            qsort(median_buffer, image_size, sizeof(double), compare_doubles);
            if (image_size % 2 == 0) {
                size_t mid = image_size / 2;
                background_value = 0.5 * (median_buffer[mid - 1] + median_buffer[mid]);
            } else {
                background_value = median_buffer[image_size / 2];
            }
        }

        for (size_t r = 0; r < height; ++r) {
            double row_sum = 0.0;
            size_t row_offset = r * width;
            for (size_t c = 0; c < width; ++c) {
                double value = image[row_offset + c];
                if (background_mean || background_median) {
                    value = fabs(value - background_value);
                }

                row_sum += value;
                column_sums[c] += value;
                total_mass += value;
            }
            y_num += (double)r * row_sum;
        }

        for (size_t c = 0; c < width; ++c) {
            x_num += (double)c * column_sums[c];
        }

        if (total_mass == 0.0) {
            if (x_out != NULL) {
                x_out[image_idx] = NAN;
            }
            if (y_out != NULL) {
                y_out[image_idx] = NAN;
            }
        } else {
            if (x_out != NULL) {
                x_out[image_idx] = x_num / total_mass;
            }
            if (y_out != NULL) {
                y_out[image_idx] = y_num / total_mass;
            }
        }
    }

    free(column_sums);
    free(median_buffer);
}
