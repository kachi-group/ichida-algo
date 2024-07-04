#include "matrix.cuh"
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
matrix* d_weights[7];
matrix* d_biases[7];
matrix** d_inputs;

int* results;

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
    matrix_mul<<<1, 1>>>(weights->data, input_layer->data, output_layer->data, weights->rows, weights->cols);
    cudaDeviceSynchronize();
    matrix_add<<<1, 1>>>(output_layer->data, biases->data, biases->rows);
    cudaDeviceSynchronize();
}

int infer(matrix* d_input) {
    matrix* outputs[2];
    outputs[0] = new_matrix_d(98, 1);
    outputs[1] = new_matrix_d(65, 1);

    propagate_fwd(d_weights[0], d_input, outputs[0], d_biases[0]);
    relu<<<1, 1>>>(outputs[0]->data, 98);
    cudaDeviceSynchronize();

    propagate_fwd(d_weights[1], outputs[0], outputs[1], d_biases[1]);
    cudaMemsetAsync(outputs[0], 0, 50 * sizeof(float));
    relu<<<1, 1>>>(outputs[1]->data, 65);
    cudaDeviceSynchronize();

    propagate_fwd(d_weights[2], outputs[1], outputs[0], d_biases[2]);
    cudaMemsetAsync(outputs[1], 0, 30 * sizeof(float));
    relu<<<1, 1>>>(outputs[0]->data, 50);
    cudaDeviceSynchronize();

    propagate_fwd(d_weights[3], outputs[0], outputs[1], d_biases[3]);
    cudaMemsetAsync(outputs[0], 0, 25 * sizeof(float));
    relu<<<1, 1>>>(outputs[1]->data, 30);
    cudaDeviceSynchronize();

    propagate_fwd(d_weights[4], outputs[1], outputs[0], d_biases[4]);
    cudaMemsetAsync(outputs[1], 0, 40 * sizeof(float));
    relu<<<1, 1>>>(outputs[0]->data, 25);
    cudaDeviceSynchronize();

    propagate_fwd(d_weights[5], outputs[0], outputs[1], d_biases[5]);
    cudaMemsetAsync(outputs[0], 0, 52 * sizeof(float));
    relu<<<1, 1>>>(outputs[1]->data, 40);
    cudaDeviceSynchronize();

    propagate_fwd(d_weights[6], outputs[1], outputs[0], d_biases[6]);
    softmax<<<1, 1>>>(outputs[0]->data, 52);
    cudaDeviceSynchronize();

    int* res_d;
    cudaMalloc(&res_d, sizeof(int));

    argmax<<<1, 1>>>(outputs[0]->data, 52, res_d);
    cudaDeviceSynchronize();

    cudaFree(outputs[0]->data);
    free(outputs[0]);
    cudaFree(outputs[1]->data);
    free(outputs[1]);

    int res_h;
    cudaMemcpy(&res_h, res_d, sizeof(int), cudaMemcpyDeviceToHost);
    return res_h;
}

void process(int input_size) {
    for (int i = 1; i <= input_size; i++) {
        results[i] = infer(d_inputs[i]);
    }
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

    d_weights[0] = get_copy(weights[0]);
    d_weights[1] = get_copy(weights[1]);
    d_weights[2] = get_copy(weights[2]);
    d_weights[3] = get_copy(weights[3]);
    d_weights[4] = get_copy(weights[4]);
    d_weights[5] = get_copy(weights[5]);
    d_weights[6] = get_copy(weights[6]);

    d_biases[0] = get_copy(biases[0]);
    d_biases[1] = get_copy(biases[1]);
    d_biases[2] = get_copy(biases[2]);
    d_biases[3] = get_copy(biases[3]);
    d_biases[4] = get_copy(biases[4]);
    d_biases[5] = get_copy(biases[5]);
    d_biases[6] = get_copy(biases[6]);

    // ------------------------------------------------------------

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

    results = (int*)malloc((size + 1) * sizeof(int));
    memset(results, 0, (size + 1) * sizeof(int));
    d_inputs = (matrix**)malloc((size + 1) * sizeof(matrix*));

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
            d_inputs[file_num] = get_copy(input);
            free(input);
        }
    }

    free(file_name);
    free(file_num_str);
    closedir(dir);

    // Process
    process(size);

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