#include "center_of_mass.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double *allocate_stack(size_t width, size_t height, size_t n_images) {
    size_t total = width * height * n_images;
    double *stack = (double *)malloc(total * sizeof(double));
    if (stack == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < total; ++i) {
        stack[i] = 1.0;
    }
    return stack;
}

static double elapsed_seconds(const struct timespec *start, const struct timespec *end) {
    double seconds = (double)(end->tv_sec - start->tv_sec);
    long nanoseconds = end->tv_nsec - start->tv_nsec;
    return seconds + (double)nanoseconds / 1e9;
}

int main(void) {
    const size_t width = 100;
    const size_t height = 100;
    const size_t n_images = 100;

    double *stack = allocate_stack(width, height, n_images);
    if (stack == NULL) {
        fprintf(stderr, "Failed to allocate stack\n");
        return EXIT_FAILURE;
    }

    double *x = (double *)malloc(n_images * sizeof(double));
    double *y = (double *)malloc(n_images * sizeof(double));
    if (x == NULL || y == NULL) {
        fprintf(stderr, "Failed to allocate output arrays\n");
        free(stack);
        free(x);
        free(y);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < 100; ++i) {
        center_of_mass(stack, width, height, n_images, "none", x, y);
    }

    struct timespec start;
    struct timespec end;

    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        perror("clock_gettime");
        free(stack);
        free(x);
        free(y);
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < 1000; ++i) {
        center_of_mass(stack, width, height, n_images, "none", x, y);
    }

    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        perror("clock_gettime");
        free(stack);
        free(x);
        free(y);
        return EXIT_FAILURE;
    }

    double elapsed = elapsed_seconds(&start, &end);
    printf("Runtime: %.6f seconds\n", elapsed);

    free(stack);
    free(x);
    free(y);

    return EXIT_SUCCESS;
}
