#pragma once

typedef float f32;
typedef unsigned char u8;

#define KERN_COLS 8
#define KERN_ROWS 4
#define SIMD_ALGN 64

typedef struct vector {
    int len;
    f32* data;
} vector;

typedef struct matrix {
    int rows;
    int cols;
    f32* data;
} matrix;

vector* new_vec_aligned(int len);

matrix* new_matrix_aligned(int rows, int cols);

// Aligned SIMD tuned matmul that assumes column major pre-transposed
void sgemv_t_tuned(const f32* weights, const f32* inputs, f32* __restrict__ results, int w_cols,
                    int w_rows);

void vector_add_inplace(int len, const f32* src, f32* dest);

void relu_inplace(f32* a, int len);

void softmax_inplace(f32* dest, int len);

void transpose_mat_inplace(matrix* in);

u8 get_max(vector* a);
