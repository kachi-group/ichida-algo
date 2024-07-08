#include "../relu.h"
void relu(float* dest, int rows) {
    for (int i = 0; i < rows; i++) {
        dest[i] = dest[i] < 0.0f ? 0.0f : dest[i];
    }
}
