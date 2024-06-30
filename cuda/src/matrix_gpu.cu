#include "../include/matrix.h"
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
__global__ void ptref(matrix* d_mat, float* d_res, int* d_cols, int* d_rows) {
    // reference newly allocated device code address to the allocated matrix
    d_mat->data = d_res;
    d_mat->cols = *d_cols;
    d_mat->rows = *d_rows;

    printf("Done2 \n");
}

// TODO: Kernel for matmul
void initmalloc(matrix* d_mat, matrix* h_mat, int rows, int cols) {
    // Allocate device memory for matrix dimensions and data
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

    // Deallocate device memory
    // cudaFree(d_cols);
    // cudaFree(d_rows);
    // cudaFree(d_res);
}

void dealloc(matrix* d_mat) {
    cudaFree(&d_mat->data);
    cudaFree(&d_mat->cols);
    cudaFree(&d_mat->rows);

    cudaFree(d_mat);
}
// Loop unrolling optimisation with a factor of 8 which should be enough to saturate a Zen3 core
__global__ void matrix_mul(matrix* weights, matrix* inputs, matrix* __restrict__ result) {

    int res_rows = result->rows;
    int w_width = weights->cols;
    // printf("width=%d", weights->cols);
    float* w_data = weights->data;
    float* i_data = inputs->data;

    int u_limit = w_width - (UNROLL_FACTOR - 1);

    for (int cur_row = 0; cur_row < res_rows; cur_row++) {
        float sum0 = 0;
        float sum1 = 0;
        float sum2 = 0;
        float sum3 = 0;
        float sum4 = 0;
        float sum5 = 0;
        float sum6 = 0;
        float sum7 = 0;
        // float sum8 = 0;
        // float sum9 = 0;
        int row_offs = cur_row * w_width;

        int k = 0;
        for (; k < u_limit; k += UNROLL_FACTOR) {
            sum0 += w_data[row_offs + k] * i_data[k];
            sum1 += w_data[row_offs + k + 1] * i_data[k + 1];
            sum2 += w_data[row_offs + k + 2] * i_data[k + 2];
            sum3 += w_data[row_offs + k + 3] * i_data[k + 3];
            sum4 += w_data[row_offs + k + 4] * i_data[k + 4];
            sum5 += w_data[row_offs + k + 5] * i_data[k + 5];
            sum6 += w_data[row_offs + k + 6] * i_data[k + 6];
            sum7 += w_data[row_offs + k + 7] * i_data[k + 7];
            // sum8 += w_data[row_offs + k + 8] * i_data[k + 8];
            // sum9 += w_data[row_offs + k + 9] * i_data[k + 9];
        }

        for (; k < w_width; k++) {
            sum0 += w_data[row_offs + k] * i_data[k];
        }

        (result->data)[cur_row] = sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7; // + sum8 + sum9;
    }
}

__global__ void matrix_add(matrix* a, matrix* b) {

    for (int i = 0; i < a->rows; i++) {
        (a->data)[i] += (b->data)[i];
    }
}

__global__ void relu(matrix* a) {
    for (int i = 0; i < a->rows; i++) {
        if ((a->data)[i] < (float)0)
            (a->data)[i] = (float)0;
    }
}

__global__ void softmax(matrix* a) {
    float res = (float)0;
    for (int i = 0; i < a->rows; i++) {
        res += exp((a->data)[i]);
    }
    for (int i = 0; i < a->rows; i++) {
        (a->data)[i] /= res;
    }
    for (int i = 0; i < a->rows; i++) {
        printf("softmax rows=%d cols =%d data=%f \n", a->rows, a->cols, (a->data)[i]);
    }
}
