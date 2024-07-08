#include "../matrix_add.h"
void matrix_add(const float* src, float* __restrict__ dest, int rows) {
    for (int i = 0; i < rows; i++) {
        dest[i] += src[i];
    }
}