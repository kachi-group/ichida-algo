#pragma once

typedef float f32;

typedef struct {
    int rows;
    int cols;
    f32* data; // array
} matrix;

__host__ __device__ matrix* new_matrix(int rows, int cols);

matrix* copy_to_device(matrix* h_mat);

matrix* new_matrix_d(int rows, int cols);

__device__ void matrix_mul(f32* a, f32* b, f32* c, int rows, int cols);

__device__ void matrix_add(f32* a, f32* b, int rows);

__device__ void relu(f32* a, int rows);

__device__ void softmax(f32* a, int rows);

__device__ int argmax(f32* a, int rows);

__device__ __host__ matrix* create_copy(matrix* mat);
