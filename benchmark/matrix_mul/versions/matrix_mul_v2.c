#include "../matrix_mul.h"

void matrix_mul(const float* weights, const float* inputs, float* __restrict__ results, int res_rows, int w_cols) {
    for (int cur_row = 0; cur_row < res_rows; cur_row++) {
        results[cur_row] = 0;
        for (int col = 0; col < w_cols; col++) {
            results[cur_row] += weights[cur_row * w_cols + col] * inputs[col];
        }
    }
}