#pragma once

typedef struct {
    int rows;
    int cols;
    float* data; // array
} matrix;

matrix* new_matrix(int rows, int cols);

matrix* copy_to_device(matrix* h_mat);

matrix* new_matrix_d(int rows, int cols);

__global__ void matrix_mul(float* a, float* b, float* c, int rows, int cols);

__global__ void matrix_add(float* a, float* b, int rows);

__global__ void relu(float* a, int rows);

__global__ void softmax(float* a, int rows);

__global__ void argmax(float* a, int rows, int* res);