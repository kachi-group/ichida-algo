#pragma once
void matrix_mul(const float* weights, const float* inputs, float* __restrict__ results, int res_rows, int w_cols);