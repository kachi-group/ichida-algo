#include "../include/matrix.h"
#include <immintrin.h> // For SIMD intrinsics
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

// Loop unrolling optimisation with a factor of 8 which should be enough to saturate a Zen3 core
void matrix_mul(const matrix* weights, const matrix* inputs, const matrix* __restrict__ result) {
    int res_rows = result->rows;
    int w_width = weights->cols;
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

// // Old version with no specific optimisation
// void matrix_mul(const matrix* __restrict__ a, const matrix* __restrict__ b, const matrix* __restrict__ result) {
//     int m = result->rows;
//     int p = a->cols;
//     for (int i = 0; i < m; i++) {
//         float sum = 0;
//         int h = i * p;
//         for (int k = 0; k < p; k++) {
//             sum += (a->data)[h + k] * ((b->data)[k]);
//         }
//         (result->data)[i] = sum;
//     }
// }

void matrix_add(matrix* a, const matrix* b) {
    int total_size = a->rows * a->cols;
    int i = 0;
    float* a_data = a->data;
    float* b_data = b->data;

    // loop unrolling
    for (; i <= total_size - UNROLL_FACTOR; i += UNROLL_FACTOR) {
        // load  data into avx register,256 is selected as it can operate float addition in simd.
        __m256 a_vec = _mm256_loadu_ps(&a_data[i]);
        __m256 b_vec = _mm256_loadu_ps(&b_data[i]);

        // adding a_vec + b_vec
        __m256 result_vec = _mm256_add_ps(a_vec, b_vec);

        // load the sum of the values back to the array
        _mm256_storeu_ps(&a_data[i], result_vec);
    }

    // Handle remaining values from the unrolled loop
    int remaining = total_size - i;

    // Masking the remaining values
    __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(remaining), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));

    // load  data into avx register,256 is selected as it can operate float addition in simd.
    __m256 a_vec = _mm256_maskload_ps(&a_data[i], mask);
    __m256 b_vec = _mm256_maskload_ps(&b_data[i], mask);

    // adding a_vec + b_vec
    __m256 result_vec = _mm256_add_ps(a_vec, b_vec);

    // load the sum of the values back to the array
    _mm256_maskstore_ps(&a_data[i], mask, result_vec);
}
void relu(matrix* a) {
    for (int i = 0; i < a->rows; i++) {
        if ((a->data)[i] < (float)0)
            (a->data)[i] = (float)0;
    }
}

void softmax(matrix* a) {
    float res = (float)0;
    for (int i = 0; i < a->rows; i++) {
        res += exp((a->data)[i]);
    }
    for (int i = 0; i < a->rows; i++) {
        (a->data)[i] /= res;
    }
}
