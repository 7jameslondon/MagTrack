#ifndef BENCHMARKS_LANGUAGE_CENTER_OF_MASS_H
#define BENCHMARKS_LANGUAGE_CENTER_OF_MASS_H

#include <stddef.h>

#ifndef CENTER_OF_MASS_API
#define CENTER_OF_MASS_API
#endif

CENTER_OF_MASS_API void center_of_mass(
    const double *stack,
    size_t width,
    size_t height,
    size_t n_images,
    const char *background,
    double *x_out,
    double *y_out);

#endif
