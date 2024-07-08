#pragma once

typedef struct {
    int rows;
    int cols;
    float* data; // array
} matrix;

double time(int n);
matrix* new_matrix_d(int rows, int cols);