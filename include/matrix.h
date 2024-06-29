#pragma once

typedef struct {
    int rows;
    int cols;
    float* data; // array
} matrix;
__global__ void built_matrix(matrix* matrix, int rows, int cols);
void initmalloc(matrix* d_mat, matrix* h_mat, int rows, int cols);
void dealloc(matrix* d_mat);

matrix* new_matrix(int rows, int cols);

void matrix_mul(const matrix* a, const matrix* b, const matrix* result);

void matrix_add(matrix* a, const matrix* b);

void relu(matrix* a);

void softmax(matrix* a);
