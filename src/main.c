#include "matrix.h"
#include "util.h"
#include "file_io.h"
#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

typedef float f32;
typedef unsigned char u8;

#define NUM_LAYERS 7

#define TENSOR_SIZE 225
#define TSIZE_ALGN_BYTES (((TENSOR_SIZE + SIMD_ALGN - 1) / SIMD_ALGN * SIMD_ALGN) * sizeof(f32))

matrix* weights[NUM_LAYERS];
vector* biases[NUM_LAYERS];

char letters[52] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                    'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                    'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z'};

void propagate_fwd(const matrix* weights, const vector* inputs, vector* results, const vector* biases) {
    sgemv_t_tuned(weights->data, inputs->data, results->data, weights->cols, weights->rows);
    // Add biases onto results
    vector_add_inplace(results->len, biases->data, results->data);
}



u8 infer(vector* input) {
    vector* outputs[NUM_LAYERS];
    outputs[0] = new_vec_aligned(98);
    outputs[1] = new_vec_aligned(65);
    outputs[2] = new_vec_aligned(50);
    outputs[3] = new_vec_aligned(30);
    outputs[4] = new_vec_aligned(25);
    outputs[5] = new_vec_aligned(40);
    outputs[6] = new_vec_aligned(52);

    propagate_fwd(weights[0], input, outputs[0], biases[0]);
    relu_inplace(outputs[0]->data, 98); 
    propagate_fwd(weights[1], outputs[0], outputs[1], biases[1]);
    relu_inplace(outputs[1]->data, 65);
    propagate_fwd(weights[2], outputs[1], outputs[2], biases[2]);
    relu_inplace(outputs[2]->data, 50);
    propagate_fwd(weights[3], outputs[2], outputs[3], biases[3]);
    relu_inplace(outputs[3]->data, 30);
    propagate_fwd(weights[4], outputs[3], outputs[4], biases[4]);
    relu_inplace(outputs[4]->data, 25);
    propagate_fwd(weights[5], outputs[4], outputs[5], biases[5]);
    relu_inplace(outputs[5]->data, 40);
    propagate_fwd(weights[6], outputs[5], outputs[6], biases[6]);
    softmax_inplace(outputs[6]->data, 52);

    u8 pred = get_max(outputs[6]);
    
    free(outputs[0]->data);
    free(outputs[0]);
    free(outputs[1]->data);
    free(outputs[1]);
    free(outputs[2]->data);
    free(outputs[2]);
    free(outputs[3]->data);
    free(outputs[3]);
    free(outputs[4]->data);
    free(outputs[4]);
    free(outputs[5]->data);
    free(outputs[5]);
    free(outputs[6]->data);
    free(outputs[6]);

    return pred;
}

u8 infer_reuse_layers(vector* input) {
    vector* outputs[NUM_LAYERS];
    outputs[0] = new_vec_aligned(98);
    outputs[1] = new_vec_aligned(65);

    propagate_fwd(weights[0], input, outputs[0], biases[0]);
    relu_inplace(outputs[0]->data, 98);
    propagate_fwd(weights[1], outputs[0], outputs[1], biases[1]);
    relu_inplace(outputs[1]->data, 65);

    outputs[0]->len = 50;
    memset(outputs[0]->data, 0, 50 * sizeof(f32));

    propagate_fwd(weights[2], outputs[1], outputs[0], biases[2]);
    relu_inplace(outputs[0]->data, 50);

    outputs[1]->len = 30;
    memset(outputs[1]->data, 0, 30 * sizeof(f32));

    propagate_fwd(weights[3], outputs[0], outputs[1], biases[3]);
    relu_inplace(outputs[1]->data, 30);

    outputs[0]->len = 25;
    memset(outputs[0]->data, 0, 25 * sizeof(f32));

    propagate_fwd(weights[4], outputs[1], outputs[0], biases[4]);
    relu_inplace(outputs[0]->data, 25);

    outputs[1]->len = 40;
    memset(outputs[1]->data, 0, 40 * sizeof(f32));

    propagate_fwd(weights[5], outputs[0], outputs[1], biases[5]);
    relu_inplace(outputs[1]->data, 40);

    outputs[0]->len = 52;
    memset(outputs[0]->data, 0, 52 * sizeof(f32));

    propagate_fwd(weights[6], outputs[1], outputs[0], biases[6]);
    softmax_inplace(outputs[0]->data, 52);

    u8 pred = get_max(outputs[0]);
    
    free(outputs[0]->data);
    free(outputs[0]);
    free(outputs[1]->data);
    free(outputs[1]);

    return pred;
}

u8 infer_reuse_input (vector* input) {
    vector* outputs[NUM_LAYERS];
    outputs[0] = new_vec_aligned(98);

    propagate_fwd(weights[0], input, outputs[0], biases[0]);
    relu_inplace(outputs[0]->data, 98);

    input->len = 65;
    memset(input->data, 0, 65 * sizeof(f32));

    propagate_fwd(weights[1], outputs[0], input, biases[1]);
    relu_inplace(input->data, 65);

    outputs[0]->len = 50;
    memset(outputs[0]->data, 0, 50 * sizeof(f32));

    propagate_fwd(weights[2], input, outputs[0], biases[2]);
    relu_inplace(outputs[0]->data, 50);

    input->len = 30;
    memset(input->data, 0, 30 * sizeof(f32));

    propagate_fwd(weights[3], outputs[0], input, biases[3]);
    relu_inplace(input->data, 30);

    outputs[0]->len = 25;
    memset(outputs[0]->data, 0, 25 * sizeof(f32));

    propagate_fwd(weights[4], input, outputs[0], biases[4]);
    relu_inplace(outputs[0]->data, 25);

    input->len = 40;
    memset(input->data, 0, 40 * sizeof(f32));

    propagate_fwd(weights[5], outputs[0], input, biases[5]);
    relu_inplace(input->data, 40);

    outputs[0]->len = 52;
    memset(outputs[0]->data, 0, 52 * sizeof(f32));

    propagate_fwd(weights[6], input, outputs[0], biases[6]);
    softmax_inplace(outputs[0]->data, 52);

    u8 pred = get_max(outputs[0]);
    
    free(outputs[0]->data);
    free(outputs[0]);

    input->len = 225;

    return pred;
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
    weights[0] = new_matrix_aligned(98, 225);
    weights[1] = new_matrix_aligned(65, 98);
    weights[2] = new_matrix_aligned(50, 65);
    weights[3] = new_matrix_aligned(30, 50);
    weights[4] = new_matrix_aligned(25, 30);
    weights[5] = new_matrix_aligned(40, 25);
    weights[6] = new_matrix_aligned(52, 40);

    biases[0] = new_vec_aligned(98);
    biases[1] = new_vec_aligned(65);
    biases[2] = new_vec_aligned(50);
    biases[3] = new_vec_aligned(30);
    biases[4] = new_vec_aligned(25);
    biases[5] = new_vec_aligned(40);
    biases[6] = new_vec_aligned(52);

    vector* input = new_vec_aligned(TENSOR_SIZE);

    read_model(weights, biases, argv[1]);

    // Transpose weights to column major
    for (int i = 0; i < NUM_LAYERS; i++) transpose_mat_inplace(weights[i]);

    const char* directory_path = argv[2];
    int input_count = file_count(directory_path);
    printf("Number of input tensors: %d\n", input_count);

    // +1 because file idx starts at 1
    u8* results = (u8*)malloc(input_count * sizeof(u8));
    f32* tensors = (f32*)aligned_alloc(SIMD_ALGN, TSIZE_ALGN_BYTES * input_count);
    
    // Read and process inputs
    char* file_path = (char*)malloc((256) * sizeof(char));
    char* file_num_str = (char*)malloc((50) * sizeof(char));

    struct dirent* entry;
    DIR* dir = opendir(directory_path);
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            strcpy(file_num_str, entry->d_name);
            file_num_str[strlen(entry->d_name) - 7] = '\0';
            int file_num = atoi(entry->d_name);
            strcpy(file_path, directory_path);
            strcat(file_path, "/");
            strcat(file_path, entry->d_name);
            read_tensor((f32*)&tensors[TSIZE_ALGN_BYTES / sizeof(f32) * (file_num - 1)], file_path);
        }
    }
    closedir(dir);
    free(file_path);
    free(file_num_str);



    // Run inference
    for (int i = 0; i < input_count; i++) {
        input->data = (f32*)&tensors[TSIZE_ALGN_BYTES / sizeof(f32) * i];
        // for (int i = 0; i < 100000; i++)
            results[i] = infer_reuse_layers(input);
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
