#pragma once

#define CUDA_CHECK(call)                                                                                               \
    cudaError_t err = call;                                                                                            \
    if (err != cudaSuccess) {                                                                                          \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(err));      \
        exit(EXIT_FAILURE);                                                                                            \
    }