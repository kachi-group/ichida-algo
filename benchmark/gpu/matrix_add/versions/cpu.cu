#include "../template.cuh"

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

void matrix_add(float* a, float* b, int rows) {
    for (int i = 0; i < rows; i++) {
        a[i] += b[i];
    }
}

double time(int n) {
    int row=100000;
    matrix* a = new_matrix(row, 1);
    matrix* b = new_matrix(row, 1);

    clock_t start = clock();
    for (int i = 0; i < n; i++) {
        matrix_add(a->data, b->data,row);
    }
    double seconds = (double)(clock() - (double)start) / CLOCKS_PER_SEC;
    return seconds;
}