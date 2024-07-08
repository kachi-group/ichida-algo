#include "../include/matrix.h"
#include "util.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

double benchmark_matrix_mul(int iterations, matrix* a, matrix* b, matrix* c) {
    double res = 0;
    for (int i = 0; i < iterations; i++) {
        uint64_t start = rdtsc();
        matrix_mul(a, b, c);
        uint64_t end = rdtsc();
        res += (double)(end - start) / (a->rows * a->cols);
    }
    res /= iterations;
    return res;
}

double benchmark_matrix_add(int iterations, matrix* a, matrix* b) {
    double res = 0;
    for (int i = 0; i < iterations; i++) {
        uint64_t start = rdtsc();
        matrix_add(a, b);
        uint64_t end = rdtsc();
        res += (double)(end - start) / (a->rows);
    }
    res /= iterations;
    return res;
}

double benchmark_relu(int iterations, matrix* a) {
    double res = 0;
    for (int i = 0; i < iterations; i++) {
        uint64_t start = rdtsc();
        relu(a);
        uint64_t end = rdtsc();
        res += (double)(end - start) / (a->rows);
    }
    res /= iterations;
    return res;
}

double benchmark_softmax(int iterations, matrix* a) {
    double res = 0;
    for (int i = 0; i < iterations; i++) {
        uint64_t start = rdtsc();
        softmax(a);
        uint64_t end = rdtsc();
        res += (double)(end - start) / (a->rows * 2);
    }
    res /= iterations;
    return res;
}

int main() {
    int iterations = 200000;
    printf("- matrix_mul: %f CPE\n",
           benchmark_matrix_mul(iterations, new_matrix(2000, 1000), new_matrix(2000, 1), new_matrix(2000, 1)));
    printf("- matrix_add: %f CPE\n", benchmark_matrix_add(iterations, new_matrix(2000, 1), new_matrix(2000, 1)));
    printf("- relu: %f CPE\n", benchmark_relu(iterations, new_matrix(2000, 1)));
    printf("- softmax: %f CPE\n", benchmark_softmax(iterations, new_matrix(2000, 1)));
}