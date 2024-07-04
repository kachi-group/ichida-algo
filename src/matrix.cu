#include "matrix.cuh"
#include "util.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define UNROLL_FACTOR 8

matrix* new_matrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (float*)malloc((rows * cols) * sizeof(float));
    return res;
}

matrix* new_matrix_d(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->cols = cols;
    cudaMalloc((void**)&(res->data), rows * cols * sizeof(float));
    return res;
}

matrix* get_copy(matrix* h_mat) {
    matrix* res = new_matrix_d(h_mat->rows, h_mat->cols);
    CUDA_CHECK(cudaMemcpy(res->data, h_mat->data, h_mat->rows * h_mat->cols * sizeof(float), cudaMemcpyHostToDevice));
    return res;
}

__global__ void matrix_mul(float* weight, float* input, float* result, int w_rows, int w_cols) {
    for (int i = 0; i < w_rows; i++) {
        float sum = 0;
        for (int j = 0; j < w_cols; j++) {
            sum += weight[i * w_cols + j] * input[j];
        }
        result[i] = sum;
    }
}

__global__ void matrix_add(float* a, float* b, int rows) {
    for (int i = 0; i < rows; i++) {
        a[i] += b[i];
    }
}

__global__ void relu(float* a, int rows) {
    for (int i = 0; i < rows; i++) {
        if ((a)[i] < (float)0)
            (a)[i] = (float)0;
    }
}

__global__ void softmax(float* a, int rows) {
    float res = (float)0;
    for (int i = 0; i < rows; i++) {
        res += exp(a[i]);
    }
    for (int i = 0; i < rows; i++) {
        a[i] /= res;
    }
}

__global__ void argmax(float* a, int rows, int* des) {
    int res = a[0];
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        if (res < a[i]) {
            res = a[i];
            idx = i;
        }
    }
    *des = idx;
}