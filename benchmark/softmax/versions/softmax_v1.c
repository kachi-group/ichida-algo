#include "../softmax.h"
#include "math.h"

void softmax(float* dest, int rows) {
    float res = 0.0f;
    for (int i = 0; i < rows; i++) {
        res += exp(dest[i]);
    }
    for (int i = 0; i < rows; i++) {
        dest[i] /= res;
    }
}