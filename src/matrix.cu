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

__global__ void ptref(matrix* d_mat, float* d_res, int* d_cols, int* d_rows) {
    d_mat->data = d_res;
    d_mat->cols = *d_cols;
    d_mat->rows = *d_rows;
}

// Allocate device memory for matrix dimensions and data
void initmalloc(matrix* d_mat, matrix* h_mat, int rows, int cols) {
    int* d_cols;
    int* d_rows;
    float* d_res;
    cudaMalloc(&d_cols, sizeof(int));
    cudaMalloc(&d_rows, sizeof(int));
    cudaMalloc(&d_res, rows * cols * sizeof(float));

    cudaMemcpy(d_rows, &rows, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cols, &cols, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_res, h_mat->data, (rows * cols * sizeof(float)), cudaMemcpyHostToDevice);

    // Call kernel to initialize the matrix structure on the device
    ptref<<<1, 1>>>(d_mat, d_res, d_cols, d_rows);
    cudaDeviceSynchronize();
}

void dealloc(matrix* d_mat) {
    cudaFree(&d_mat->data);
    cudaFree(&d_mat->cols);
    cudaFree(&d_mat->rows);

    cudaFree(d_mat);
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