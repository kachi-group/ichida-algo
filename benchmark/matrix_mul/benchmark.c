#include "matrix_mul.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __ARM_ARCH_ISA_A64
uint64_t rdtsc() {
    uint64_t val;
    __asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

#else
#include <x86intrin.h>
uint64_t rdtsc() { return __rdtsc(); }
#endif

typedef struct {
    float* data;
    int rows;
    int cols;
} matrix;

matrix* new_matrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (float*)malloc((rows * cols) * sizeof(float));
    return res;
}

int main(int argc, char* argv[]) {
    long n = 0;
    if (argc > 1) {
        n = atol(argv[1]);
    } else {
        printf("Error!");
        exit(1);
    }
    clock_t start = clock();
    matrix* a = new_matrix(98, 255);
    matrix* b = new_matrix(255, 1);
    matrix* c = new_matrix(98, 1);
    for (int i = 0; i < n; i++) {
        matrix_mul(a->data, b->data, c->data, c->rows, a->cols);
    }
    float seconds = (float)(clock() - (float)start) / CLOCKS_PER_SEC;
    printf("%f", seconds);
}