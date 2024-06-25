#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    int rows;
    int cols;
    float** data;
} matrix;

matrix* createMatrix(int rows, int cols);

void multiplyMatrices(const matrix* a, const matrix* b, const matrix* result);

void addMatrix(matrix* a, const matrix* b);

void ReLU(matrix* a);

void softmax(matrix* a);

#endif