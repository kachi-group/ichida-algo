#include "relu.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    float* data;
    int rows;
    int cols;
} matrix;

matrix* new_matrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (float*)malloc((rows * cols) * sizeof(float));
    return res;
}

int main(int argc, char* argv[]) {
    long n = 0;
    if (argc > 1) {
        n = atol(argv[1]);
    } else {
        printf("Error!");
        exit(1);
    }
    clock_t start = clock();
    matrix* a = new_matrix(255, 1);
    for (int i = 0; i < n; i++) {
        relu(a->data, a->rows);
    }
    float seconds = (float)(clock() - (float)start) / CLOCKS_PER_SEC;
    printf("%f", seconds);
}