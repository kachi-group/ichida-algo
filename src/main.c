#include "../include/matrix.h"

#define _DEFAULT_SOURCE // Fixes
#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define NUM_LAYERS 7
#define TENSOR_SIZE 225

typedef uint8_t u8;
typedef float f32;

matrix* weights[NUM_LAYERS];
matrix* biases[NUM_LAYERS];

static char letters[52] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
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

void propagate_fwd(const matrix* weights, const matrix* inputs, matrix* outputs, const matrix* biases) {
    matrix_mul(inputs->data, weights->data, outputs->data, outputs->rows, weights->cols);
    matrix_add_inplace_1d(biases->data, inputs->data, inputs->rows);
}

// Get result from output layer
u8 get_max(float* src, int rows) {
    int idx = 0;
    float res = src[0];
    for (int i = 0; i < rows; i++) {
        if (res < src[i]) {
            res = src[i];
            idx = i;
        }
    }
    return idx;
}

u8 infer(matrix* input) {
    matrix* mdl_layers[NUM_LAYERS];
    mdl_layers[0] = new_matrix(98, 1);
    mdl_layers[1] = new_matrix(65, 1);
    mdl_layers[2] = new_matrix(50, 1);
    mdl_layers[3] = new_matrix(30, 1);
    mdl_layers[4] = new_matrix(25, 1);
    mdl_layers[5] = new_matrix(40, 1);
    mdl_layers[6] = new_matrix(52, 1);

    propagate_fwd(weights[0], input, mdl_layers[0], biases[0]);
    relu(mdl_layers[0]->data, mdl_layers[0]->rows);

    propagate_fwd(weights[1], mdl_layers[0], mdl_layers[1], biases[1]);
    relu(mdl_layers[1]->data, mdl_layers[1]->rows);

    propagate_fwd(weights[2], mdl_layers[1], mdl_layers[2], biases[2]);
    relu(mdl_layers[2]->data, mdl_layers[2]->rows);

    propagate_fwd(weights[3], mdl_layers[2], mdl_layers[3], biases[3]);
    relu(mdl_layers[3]->data, mdl_layers[3]->rows);

    propagate_fwd(weights[4], mdl_layers[3], mdl_layers[4], biases[4]);
    relu(mdl_layers[4]->data, mdl_layers[4]->rows);

    propagate_fwd(weights[5], mdl_layers[4], mdl_layers[5], biases[5]);
    relu(mdl_layers[5]->data, mdl_layers[5]->rows);

    propagate_fwd(weights[6], mdl_layers[5], mdl_layers[6], biases[6]);
    softmax(mdl_layers[6]->data, mdl_layers[6]->rows);

    return get_max(mdl_layers[6]->data, mdl_layers[6]->rows);
}

int file_count(const char* dir_path) {
    struct dirent* entry;
    DIR* dir = opendir(dir_path);

    // Count inputs
    int num_inputs = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG)
            num_inputs++;
    }

    return num_inputs;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Not enough arguments. Usage: speed_cpu <path_to_model.txt> <tensors_dir/>");
        return EXIT_FAILURE;
    }

    // Start timing
    struct timeval stop, start;
    gettimeofday(&start, NULL);

    // Dimensions of target model are hardcoded
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

    // Holding place for input tensors
    matrix* input = new_matrix(225, 1);

    // ---------------------------------------------------------------------------------- Read from dir

    const char* directory_path = argv[2];
    int input_count = file_count(directory_path);
    printf("Number of input tensors: %d\n", input_count);

    // +1 because file idx starts at 1
    u8* results = (u8*)malloc(input_count * sizeof(u8));

    __attribute__((aligned(32))) f32* tensors = (f32*)malloc(sizeof(f32) * TENSOR_SIZE * input_count);

    char* file_path = (char*)malloc((256) * sizeof(char));
    char* file_num_str = (char*)malloc((50) * sizeof(char));

    // Read all tensors into tensors arr
    DIR* dir = opendir(directory_path);
    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            // Get input number
            strcpy(file_num_str, entry->d_name);
            file_num_str[strlen(entry->d_name) - 7] = '\0';
            int file_num = atoi(entry->d_name);

            // Get full path to file
            strcpy(file_path, directory_path);
            strcat(file_path, "/");
            strcat(file_path, entry->d_name);

            // Read tensor into full array
            FILE* file = fopen(file_path, "rb");

            // Offset into correct sector of array
            fread(tensors + ((file_num - 1) * TENSOR_SIZE), TENSOR_SIZE, sizeof(f32), file);
            fclose(file);
        }
    }
    free(file_path);
    free(file_num_str);
    closedir(dir);

    for (int i = 0; i < input_count; i++) {
        input->data = tensors + (i * TENSOR_SIZE);
        results[i] = infer(input);
    }

    // Write to csv file
    FILE* csv_file = fopen("results.csv", "w+");
    fprintf(csv_file, "image_number, guess\n");
    for (int i = 0; i < input_count; i++) {
        fprintf(csv_file, "%d, %c\n", i + 1, letters[results[i]]);
    }
    fclose(csv_file);

    // Time taken
    gettimeofday(&stop, NULL);
    printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

    return EXIT_SUCCESS;
}
