#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

// Aligned to cache line and kernel size
matrix* new_matrix_aligned(int rows, int cols) {
    // Pad width to fit kernel
    int kern_align_f32 = (rows * cols + KERN_COLS - 1) / KERN_COLS * KERN_COLS;

    matrix* new_mat = (matrix*)malloc(sizeof(matrix));
    new_mat->rows = rows;
    new_mat->cols = cols;

    // Align entire array for simd access and better cache line utilisation
    new_mat->data = (f32*)aligned_alloc(
        SIMD_ALIGN, (((kern_align_f32 * sizeof(f32)) + SIMD_ALIGN_F32 - 1) / SIMD_ALIGN_F32 * SIMD_ALIGN_F32));

    return new_mat;
}

// Aligned to cache line and kernel size
vector* new_vec_aligned(int len) {
    // Pad width to fit kernel
    int kern_align_f32 = (len + KERN_COLS - 1) / KERN_COLS * KERN_COLS;

    vector* new_vec = (vector*)malloc(sizeof(vector));
    new_vec->len = len;

    // Align entire array for simd access and better cache line utilisation
    new_vec->data = (f32*)aligned_alloc(
        SIMD_ALIGN, (((kern_align_f32 * sizeof(f32)) + SIMD_ALIGN_F32 - 1) / SIMD_ALIGN_F32 * SIMD_ALIGN_F32));

    memset(new_vec->data, 0, kern_align_f32 * sizeof(f32));

    return new_vec;
}

// ver. Artemis Rosman simd_intrin 2x8
static void kernel(const float* in, const float* wg, float* rs, int start_row, int start_col, int w_width) {
    // printf("Kernel at row %d col %d\n", start_row, start_col);
    __m256 res = _mm256_load_ps(&rs[start_col]);

    for (int row = 0; row < KERN_ROWS; row++) {
        __m256 wr = _mm256_load_ps(&wg[w_width * (start_row + row) + start_col]);
        __m256 is = _mm256_set1_ps(in[start_row + row]);
        res = _mm256_fmadd_ps(wr, is, res);
    }
    _mm256_store_ps(&rs[start_col], res);
}

// Ver. Artemis Rosman
// W rows and W width is expected to be for the column major matrix, i.e. len of
// in vec = w_rows, len of out vec = w_cols
void sgemv_t_tuned(const float* weights, const float* inputs, float* __restrict__ results, int w_width, int w_rows) {
    // Perform mult using kernel
    for (int row = 0; row < w_rows; row += KERN_ROWS) {
        for (int col = 0; col < w_width; col += KERN_COLS) {
            kernel(inputs, weights, results, row, col, w_width);
        }
    }
}

// TODO: SIMD tuned versions if these are a noticeable impact
void vector_add_inplace(int len, const f32* src, f32* __restrict__ dest) {
    for (int i = 0; i < len; i++) {
        dest[i] += src[i];
    }
}

void relu_inplace(f32* dest, int len) {
    for (int i = 0; i < len; i++) {
        dest[i] = dest[i] < 0.0f ? 0.0f : dest[i];
    }
}

// Hacky but fast and accurate for existing inputs
static inline float fastexp(float x) {
    int tmp = (int)(1512775 * x + 1072632447);
    float result;
    memcpy(&result, &tmp, sizeof(result));
    return result;
}

void softmax_inplace(f32* dest, int len) {
    float res = 0.0f;
    for (int i = 0; i < len; i++) {
        res += fastexp(dest[i]);
    }
    for (int i = 0; i < len; i++) {
        dest[i] /= res;
    }
}

// Get result from output layer
u8 argmax(f32* in, int len) {
    int idx = 0;
    float res = in[0];
    for (int i = 0; i < len; i++) {
        if (res < in[i]) {
            res = in[i];
            idx = i;
        }
    }
    return idx;
}

void transpose_mat_inplace(matrix* in) {
    int cols_before = in->cols;
    int rows_before = in->rows;

    // Swapped for transpose
    int pad_w_rows = (cols_before + KERN_ROWS - 1) / KERN_ROWS * KERN_ROWS;
    int pad_w_width = (rows_before + KERN_COLS - 1) / KERN_COLS * KERN_COLS;
    f32* transposed = (f32*)aligned_alloc(SIMD_ALIGN, (((pad_w_rows * pad_w_width * sizeof(f32)) + SIMD_ALIGN_F32 - 1) /
                                                       SIMD_ALIGN_F32 * SIMD_ALIGN_F32));
    memset(transposed, 0, pad_w_rows * pad_w_width * sizeof(f32));

    for (int row = 0; row < rows_before; row++) {
        for (int col = 0; col < cols_before; col++) {
            transposed[col * pad_w_width + row] = in->data[row * cols_before + col];
        }
    }

    free(in->data);
    in->data = transposed;
    // Swap dims
    in->cols = pad_w_width;
    in->rows = cols_before;
}
