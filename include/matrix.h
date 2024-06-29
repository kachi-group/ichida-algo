#pragma once

typedef struct {
    int rows;
    int cols;
    float* data; // array
} matrix;
void initmalloc(matrix* d_mat, matrix* h_mat, int rows, int cols);
void dealloc(matrix* d_mat);

matrix* new_matrix(int rows, int cols);

__global__ void matrix_mul(matrix* a, matrix* b, matrix* result);

__global__ void matrix_add(matrix* a, matrix* b);

__global__ void relu(matrix* a);

__global__ void softmax(matrix* a);
