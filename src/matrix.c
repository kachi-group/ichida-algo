#include "matrix.h"
#include "math.h"
#include "stdio.h"
#include "stdlib.h"

matrix* createMatrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        res->data[i] = (float*)malloc(cols * sizeof(float));
    }
    return res;
}

void multiplyMatrices(const matrix* a, const matrix* b, const matrix* result) {
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += (a->data)[i][k] * ((b->data)[k][j]);
            }
            (result->data)[i][j] = sum;
        }
    }
}

void addMatrix(matrix* a, const matrix* b) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            (a->data)[i][j] += (b->data)[i][j];
        }
    }
}

void ReLU(matrix* a) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            if ((a->data)[i][j] < (float)0)
                (a->data)[i][j] = (float)0;
        }
    }
}

void softmax(matrix* a) {
    float res = (float)0;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            res += exp((a->data)[i][j]);
        }
    }
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            (a->data)[i][j] /= res;
        }
    }
}
