#include "../include/matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

matrix* new_matrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (float*)malloc((rows * cols) * sizeof(float));
    return res;
}

matrix* new_matrix_zero(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));

    res->rows = rows;
    res->cols = cols;
    res->data = (float*)malloc((rows * cols) * sizeof(float));
    return res;
}

// Turns out the compiler was doing most of the same work so we can reduce this in complexity significantly.
void matrix_mul(const float* inputs, const float* weights, float* __restrict__ results, int res_rows, int w_cols) {
    for (int cur_row = 0; cur_row < res_rows; cur_row++) {
        results[cur_row] = 0;
        for (int col = 0; col < w_cols; col++) {
            results[cur_row] += weights[cur_row * w_cols + col] * inputs[col];
        }
    }
}

void matrix_add_inplace_1d(const float* src, float* __restrict__ dest, int rows) {
    for (int i = 0; i < rows; i++) {
        dest[i] += src[i];
    }
}

void relu(float* dest, int rows) {
    for (int i = 0; i < rows; i++) {
        dest[i] = dest[i] < 0.0f ? 0.0f : dest[i];
    }
}

void softmax(float* dest, int rows) {
    float res = 0.0f;
    for (int i = 0; i < rows; i++) {
        res += exp(dest[i]);
    }
    for (int i = 0; i < rows; i++) {
        dest[i] /= res;
    }
}
