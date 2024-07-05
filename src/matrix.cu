#include "matrix.cuh"
#include "util.cuh"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define UNROLL_FACTOR 8

__host__ __device__ matrix* new_matrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (float*)malloc((rows * cols) * sizeof(float));
    return res;
}

__global__ void alloc(matrix* res,float* data,int rows,int cols){
    res->rows=rows;
    res->cols=cols;
    res->data=data;
}

matrix* new_matrix_d(int rows, int cols) {
    matrix* res;
    CUDA_CHECK(cudaMalloc(&res,sizeof(matrix)));
    float* data;
    cudaMalloc(&data,rows*cols*sizeof(float));
    alloc<<<1,1>>>(res,data,rows,cols);
    cudaDeviceSynchronize();
    return res;
}

matrix* copy_to_device(matrix* h_mat) {
    matrix* res ;
    CUDA_CHECK(cudaMalloc(&res,sizeof(matrix)));
    float* data;
    cudaMalloc(&data,h_mat->rows*h_mat->cols*sizeof(float));
    cudaMemcpy(data, h_mat->data, h_mat->rows * h_mat->cols * sizeof(float), cudaMemcpyHostToDevice);
    alloc<<<1,1>>>(res,data,h_mat->rows,h_mat->cols);
    return res;
}

__device__ __host__ matrix* create_copy(matrix* mat){
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = mat->rows;
    res->cols = mat->cols;
    res->data = (float*)malloc((res->rows * res->cols) * sizeof(float));
    memcpy(res->data,mat->data,res->rows*res->cols*sizeof(float));
    return res;
}

__device__ void matrix_mul(float* weight, float* input, float* result, int w_rows, int w_cols) {
    for (int i = 0; i < w_rows; i++) {
        float sum = 0;
        for (int j = 0; j < w_cols; j++) {
            sum += weight[i * w_cols + j] * input[j];
        }
        result[i] = sum;
    }
}

__device__ void matrix_add(float* a, float* b, int rows) {
    for (int i = 0; i < rows; i++) {
        a[i] += b[i];
    }
}

__device__ void relu(float* a, int rows) {
    for (int i = 0; i < rows; i++) {
        if ((a)[i] < (float)0)
            (a)[i] = (float)0;
    }
}

__device__ void softmax(float* a, int rows) {
    float res = (float)0;
    for (int i = 0; i < rows; i++) {
        res += exp(a[i]);
    }
    for (int i = 0; i < rows; i++) {
        a[i] /= res;
    }
}

__device__ int argmax(float* a, int rows) {
    int res = a[0];
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        if (res < a[i]) {
            res = a[i];
            idx = i;
        }
    }
    return idx;
}