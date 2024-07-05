#pragma once

typedef struct {
    int rows;
    int cols;
    float* data; // array
} matrix;

 __host__ __device__ matrix* new_matrix(int rows, int cols);

matrix* copy_to_device(matrix* h_mat);

matrix* new_matrix_d(int rows, int cols);

__device__ void matrix_mul(float* a, float* b, float* c, int rows, int cols);

__device__ void matrix_add(float* a, float* b, int rows);

__device__ void relu(float* a, int rows);

__device__ void softmax(float* a, int rows);

__device__ int argmax(float* a, int rows);