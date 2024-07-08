#include "matrix.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define UNROLL_FACTOR 8

__host__ __device__ matrix* new_matrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (f32*)malloc((rows * cols) * sizeof(f32));
    return res;
}

__global__ void alloc(matrix* res, f32* data, int rows, int cols) {
    res->rows = rows;
    res->cols = cols;
    res->data = data;
}

matrix* new_matrix_d(int rows, int cols) {
    matrix* res;
    cudaMalloc(&res, sizeof(matrix));
    f32* data;
    cudaMalloc(&data, rows * cols * sizeof(f32));
    alloc<<<1, 1>>>(res, data, rows, cols);
    return res;
}

matrix* copy_to_device(matrix* h_mat) {
    matrix* res;
    cudaMalloc(&res, sizeof(matrix));
    f32* data;
    cudaMalloc(&data, h_mat->rows * h_mat->cols * sizeof(f32));
    cudaMemcpy(data, h_mat->data, h_mat->rows * h_mat->cols * sizeof(f32), cudaMemcpyHostToDevice);
    alloc<<<1, 1>>>(res, data, h_mat->rows, h_mat->cols);
    return res;
}

__device__ __host__ matrix* create_copy(matrix* mat) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = mat->rows;
    res->cols = mat->cols;
    res->data = (f32*)malloc((res->rows * res->cols) * sizeof(f32));
    memcpy(res->data, mat->data, res->rows * res->cols * sizeof(f32));
    return res;
}

__device__ void matrix_mul(f32* weight, f32* input, f32* result, int w_rows, int w_cols) {
    for (int i = 0; i < w_rows; i++) {
        f32 sum = 0;
        int j = 0;

        for (; j <= w_cols - 4; j += 4) {
            sum += weight[i * w_cols + j] * input[j];
            sum += weight[i * w_cols + j + 1] * input[j + 1];
            sum += weight[i * w_cols + j + 2] * input[j + 2];
            sum += weight[i * w_cols + j + 3] * input[j + 3];
        }
        for (; j < w_cols; j++) {
            sum += weight[i * w_cols + j] * input[j];
        }
        result[i] = sum;
    }
}

__device__ void matrix_add(f32* a, f32* b, int rows) {
    for (int i = 0; i < rows; i++) {
        a[i] += b[i];
    }
}

__device__ void relu(f32* a, int rows) {
    for (int i = 0; i < rows; i++) {
        a[i] = (a[i] > 0) ? a[i] : 0;
    }
}

__device__ void softmax(f32* a, int rows) {
    f32 sum = 0.0;
    for (size_t i = 0; i < rows; i++) {
        sum += __expf(a[i]);
    }
    f32 t = __logf(sum);
    for (size_t i = 0; i < rows; i++) {
        a[i] = __expf(a[i] - t);
    }
}

__device__ int argmax(f32* a, int rows) {
    f32 res = a[0];
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        if (res < a[i]) {
            res = a[i];
            idx = i;
        }
    }
    return idx;
}
