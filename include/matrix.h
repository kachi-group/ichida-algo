#pragma once

typedef struct {
    int rows;
    int cols;
    float* data __attribute__((aligned(32)));
} matrix;

matrix* new_matrix(int rows, int cols);

void matrix_mul(const float* inputs, const float* weights, float* __restrict__ results, int res_rows, int w_cols);

void matrix_add_inplace_1d(const float* src, float* __restrict__ dest, int rows);

void relu(float* dest, int rows);

void softmax(float* dest, int rows);
