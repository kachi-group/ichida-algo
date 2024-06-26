#pragma once

typedef struct {
    int rows;
    int cols;
    float* data;
} matrix;

matrix* new_matrix(int rows, int cols);

void matrix_mul(const matrix* a, const matrix* b, const matrix* result);

void matrix_add(matrix* a, const matrix* b);

void relu(matrix* a);

void softmax(matrix* a);
