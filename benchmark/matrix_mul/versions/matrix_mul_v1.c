#include "../matrix_mul.h"

void matrix_mul(const float* weights, const float* inputs, float* __restrict__ results, int res_rows, int w_cols) {
    for (int i = 0; i < res_rows; i++) {
        float sum = 0;
        int h = i * w_cols;
        for (int k = 0; k < w_cols; k++) {
            sum += weights[h + k] * inputs[k];
        }
        results[i] = sum;
    }
}