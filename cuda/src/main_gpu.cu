#include "../include/matrix.h"
#include <dirent.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define NUM_LAYERS 7

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t err = call;                                                                                        \
        if (err != cudaSuccess) {                                                                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", __func__, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

matrix* weights[NUM_LAYERS];
matrix* biases[NUM_LAYERS];

// device weights and biases;
matrix* d_weights;
matrix* d_biases;
matrix* d_input;
// allocating device matrix for weights and biases
// CUDA_CHECK(cudaMalloc(&d_weights, NUM_LAYERS * sizeof(matrix)));
// CUDA_CHECK(cudaMalloc(&d_biases, NUM_LAYERS * sizeof(matrix)));

char letters[52] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                    'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                    'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z'};

void process_weights_str(char* line, int layer) {
    char* token;
    float value;
    const char* delimiter = ",";

    token = strtok(line, delimiter);
    int n = (weights[layer]->rows) * (weights[layer]->cols);
    for (int i = 0; i < n; i++) {
        value = strtof(token, NULL);
        (weights[layer]->data)[i] = value;
        token = strtok(NULL, delimiter);
    }
}

void process_biases_str(char* line, int layer) {
    char* token;
    float value;
    const char* delimiter = ",";

    token = strtok(line, delimiter);

    int n = biases[layer]->rows;
    for (int i = 0; i < n; i++) {
        value = strtof(token, NULL);
        (biases[layer]->data)[i] = value;
        token = strtok(NULL, delimiter);
    }
}

void read_model(const char* file_name) {
    FILE* file = fopen(file_name, "r");

    char* line = NULL;
    size_t len = 0;
    int line_number = 0;
    int layer = 0;

    while ((getline(&line, &len, file)) != -1) {
        if ((line_number - 1) % 4 == 0) {
            process_weights_str(line, layer);
        } else if ((line_number - 3) % 4 == 0) {
            process_biases_str(line, layer);
            layer++;
        }
        line_number++;
    }

    free(line);
    fclose(file);
}

void read_tensor(matrix* a, const char* fileName) {
    FILE* file = fopen(fileName, "r");
    char* line = NULL;
    size_t len = 0;

    getline(&line, &len, file);
    char* token;
    float value;
    const char* delimiter = ",";
    token = strtok(line, delimiter);

    for (int i = 0; i < 225; i++) {
        value = strtof(token, NULL);
        (a->data)[i] = value;
        token = strtok(NULL, delimiter);
    }
    free(line);
    fclose(file);
}

void propagate_fwd(matrix* weights, matrix* input_layer, matrix* output_layer, matrix* biases) {
    // everything here is device code
    // matrix_mul(weights, input_layer, output_layer);
    matrix_mul<<<1, 1>>>(weights, input_layer, output_layer);
    cudaDeviceSynchronize();
    matrix_add<<<1, 1>>>(output_layer, biases);
    cudaDeviceSynchronize();
}

// Get result from output layer
__global__ void get_max(matrix* a, int* d_int) {
    int idx = 0;
    float res = a->data[0];
    for (int i = 0; i < a->rows; i++) {
        if (res < a->data[i]) {
            res = a->data[i];
            idx = i;
        }
    }
    *d_int = idx;
}

int infer(matrix* d_input) {
    matrix* mdl_layers[NUM_LAYERS];    // host
    matrix* d_mdl_layers;              // device
    mdl_layers[0] = new_matrix(98, 1); // you may see garbage values as it is unitialized
    mdl_layers[1] = new_matrix(65, 1);
    mdl_layers[2] = new_matrix(50, 1);
    mdl_layers[3] = new_matrix(30, 1);
    mdl_layers[4] = new_matrix(25, 1);
    mdl_layers[5] = new_matrix(40, 1);
    mdl_layers[6] = new_matrix(52, 1);

    CUDA_CHECK(cudaMalloc(&d_mdl_layers, NUM_LAYERS * sizeof(matrix)));
    initmalloc(&d_mdl_layers[0], mdl_layers[0], 98, 1);
    initmalloc(&d_mdl_layers[1], mdl_layers[1], 65, 1);
    initmalloc(&d_mdl_layers[2], mdl_layers[2], 50, 1);
    initmalloc(&d_mdl_layers[3], mdl_layers[3], 30, 1);
    initmalloc(&d_mdl_layers[4], mdl_layers[4], 25, 1);
    initmalloc(&d_mdl_layers[5], mdl_layers[5], 40, 1);
    initmalloc(&d_mdl_layers[6], mdl_layers[6], 52, 1);

    propagate_fwd(&d_weights[0], d_input, &d_mdl_layers[0], &d_biases[0]);
    relu<<<1, 1>>>(&d_mdl_layers[0]);
    cudaDeviceSynchronize();

    propagate_fwd(&d_weights[1], &d_mdl_layers[0], &d_mdl_layers[1], &d_biases[1]);
    relu<<<1, 1>>>(&d_mdl_layers[1]);
    cudaDeviceSynchronize();

    propagate_fwd(&d_weights[2], &d_mdl_layers[1], &d_mdl_layers[2], &d_biases[2]);
    relu<<<1, 1>>>(&d_mdl_layers[2]);
    cudaDeviceSynchronize();

    propagate_fwd(&d_weights[3], &d_mdl_layers[2], &d_mdl_layers[3], &d_biases[3]);
    relu<<<1, 1>>>(&d_mdl_layers[3]);
    cudaDeviceSynchronize();

    propagate_fwd(&d_weights[4], &d_mdl_layers[3], &d_mdl_layers[4], &d_biases[4]);
    relu<<<1, 1>>>(&d_mdl_layers[4]);
    cudaDeviceSynchronize();

    propagate_fwd(&d_weights[5], &d_mdl_layers[4], &d_mdl_layers[5], &d_biases[5]);
    relu<<<1, 1>>>(&d_mdl_layers[5]);
    cudaDeviceSynchronize();

    propagate_fwd(&d_weights[6], &d_mdl_layers[5], &d_mdl_layers[6], &d_biases[6]);
    softmax<<<1, 1>>>(&d_mdl_layers[6]);
    cudaDeviceSynchronize();

    int* d_int;
    int h_int = 0;

    CUDA_CHECK(cudaMalloc((void**)&d_int, sizeof(int)));
    get_max<<<1, 1>>>(&d_mdl_layers[6], d_int);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(&h_int, d_int, sizeof(int), cudaMemcpyDeviceToHost));

    dealloc(&d_mdl_layers[0]);
    dealloc(&d_mdl_layers[1]);
    dealloc(&d_mdl_layers[2]);
    dealloc(&d_mdl_layers[3]);
    dealloc(&d_mdl_layers[4]);
    dealloc(&d_mdl_layers[5]);
    dealloc(&d_mdl_layers[6]);

    return h_int;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Not enough arguments.");
        return EXIT_FAILURE;
    }

    // Start timing
    struct timeval stop, start;
    gettimeofday(&start, NULL);

    // TODO: find a way to load static weights and biases
    // Load model (The memory of those code should be initialize during compile time to enchance the speed)
    weights[0] = new_matrix(98, 225);
    weights[1] = new_matrix(65, 98);
    weights[2] = new_matrix(50, 65);
    weights[3] = new_matrix(30, 50);
    weights[4] = new_matrix(25, 30);
    weights[5] = new_matrix(40, 25);
    weights[6] = new_matrix(52, 40);

    biases[0] = new_matrix(98, 1);
    biases[1] = new_matrix(65, 1);
    biases[2] = new_matrix(50, 1);
    biases[3] = new_matrix(30, 1);
    biases[4] = new_matrix(25, 1);
    biases[5] = new_matrix(40, 1);
    biases[6] = new_matrix(52, 1);

    read_model(argv[1]);
    // initialize d_weights struct matrix arr;
    CUDA_CHECK(cudaMalloc(&d_weights, NUM_LAYERS * sizeof(matrix)));
    CUDA_CHECK(cudaMalloc(&d_biases, NUM_LAYERS * sizeof(matrix)));
    initmalloc(&d_weights[0], weights[0], 98, 225);
    initmalloc(&d_weights[1], weights[1], 65, 98);
    initmalloc(&d_weights[2], weights[2], 50, 65);
    initmalloc(&d_weights[3], weights[3], 30, 50);
    initmalloc(&d_weights[4], weights[4], 25, 30);
    initmalloc(&d_weights[5], weights[5], 40, 25);
    initmalloc(&d_weights[6], weights[6], 52, 40);
    initmalloc(&d_biases[0], biases[0], 98, 1);
    initmalloc(&d_biases[1], biases[1], 65, 1);
    initmalloc(&d_biases[2], biases[2], 50, 1);
    initmalloc(&d_biases[3], biases[3], 30, 1);
    initmalloc(&d_biases[4], biases[4], 25, 1);
    initmalloc(&d_biases[5], biases[5], 40, 1);
    initmalloc(&d_biases[6], biases[6], 52, 1);

    // Run program
    const char* directory_path = argv[2];
    struct dirent* entry;
    DIR* dir = opendir(directory_path);

    // Read and process inputs
    char* file_name = (char*)malloc((100) * sizeof(char));
    char* file_num_str = (char*)malloc((100) * sizeof(char));

    int file_num;
    int size = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            size++;
        }
    }
    int* results = (int*)malloc((size + 1) * sizeof(int));
    dir = opendir(directory_path);
    matrix* d_input;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            matrix* input = new_matrix(225, 1);
            strcpy(file_num_str, entry->d_name);
            file_num_str[strlen(entry->d_name) - 7] = '\0';
            file_num = atoi(entry->d_name);
            strcpy(file_name, directory_path);
            strcat(file_name, "/");
            strcat(file_name, entry->d_name);
            read_tensor(input, file_name);
            CUDA_CHECK(cudaMalloc(&d_input, 255 * sizeof(matrix)));
            initmalloc(d_input, input, 1, 225);
            results[file_num] = infer(d_input);
            dealloc(d_input);

            free(input);
        }
    }

    free(file_name);
    free(file_num_str);
    closedir(dir);

    // Write to csv file
    FILE* csv_file = fopen("results.csv", "w+");
    fprintf(csv_file, "image_number, guess\n");
    for (int i = 1; i <= size; i++) {
        fprintf(csv_file, "%d, %c\n", i, letters[results[i]]);
    }
    fclose(csv_file);

    // Time taken
    gettimeofday(&stop, NULL);
    printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

    return EXIT_SUCCESS;
}