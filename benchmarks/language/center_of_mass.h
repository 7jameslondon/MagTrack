#ifndef BENCHMARKS_LANGUAGE_CENTER_OF_MASS_H
#define BENCHMARKS_LANGUAGE_CENTER_OF_MASS_H

#include <stddef.h>

void center_of_mass(
    const double *stack,
    size_t width,
    size_t height,
    size_t n_images,
    const char *background,
    double *x_out,
    double *y_out);

#endif
