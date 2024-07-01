#include "../include/matrix.h"
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define UNROLL_FACTOR 8

matrix* new_matrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (float*)malloc((rows * cols) * sizeof(float));
    return res;
}

// Loop unrolling optimisation with a factor of 8 which should be enough to saturate a Zen3 core
void matrix_mul(const matrix* weights, const matrix* inputs, const matrix* __restrict__ result) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, weights->rows, weights->cols, 1.0f, weights->data, weights->cols,
                inputs->data, 1, 0.0f, result->data, 1);
}

// // Old version with no specific optimisation
// void matrix_mul(const matrix* __restrict__ a, const matrix* __restrict__ b, const matrix* __restrict__ result) {
//     int m = result->rows;
//     int p = a->cols;
//     for (int i = 0; i < m; i++) {
//         float sum = 0;
//         int h = i * p;
//         for (int k = 0; k < p; k++) {
//             sum += (a->data)[h + k] * ((b->data)[k]);
//         }
//         (result->data)[i] = sum;
//     }
// }

void matrix_add(matrix* a, const matrix* b) {
    for (int i = 0; i < a->rows; i++) {
        (a->data)[i] += (b->data)[i];
    }
}

void relu(matrix* a) {
    for (int i = 0; i < a->rows; i++) {
        if ((a->data)[i] < (float)0)
            (a->data)[i] = (float)0;
    }
}

void softmax(matrix* a) {
    float res = (float)0;
    for (int i = 0; i < a->rows; i++) {
        res += exp((a->data)[i]);
    }
    for (int i = 0; i < a->rows; i++) {
        (a->data)[i] /= res;
    }
}
