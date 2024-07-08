#pragma once

#include <stdio.h>

void prntmat(float* mat, int rows, int cols) {

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}