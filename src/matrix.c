#include "../include/matrix.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

matrix* new_matrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (float*)malloc((rows * cols) * sizeof(float));
    return res;
}

void matrix_mul(const matrix* __restrict__ a, const matrix* __restrict__ b, const matrix* __restrict__ result) {
    int m = result->rows;
    int p = a->cols;
    for (int i = 0; i < m; i++) {
        float sum = 0;
        int h = i * p;
        for (int k = 0; k < p; k++) {
            sum += (a->data)[h + k] * ((b->data)[k]);
        }
        (result->data)[i] = sum;
    }
}

void matrix_add(matrix* a, const matrix* b) {
    for (int i = 0; i < a->rows; i++) {
        (a->data)[i] += (b->data)[i];
    }
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
