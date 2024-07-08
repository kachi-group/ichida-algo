#include "file_io.h"
#include "matrix.h"
#include "util.h"
#include <dirent.h>
#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

typedef float f32;
typedef unsigned char u8;

#define NUM_LAYERS 7

#define TENSOR_SIZE 225
#define TSIZE_ALIGN_BYTES (((TENSOR_SIZE + SIMD_ALIGN_F32 - 1) / SIMD_ALIGN_F32 * SIMD_ALIGN_F32) * sizeof(f32))

matrix* weights[NUM_LAYERS];
vector* biases[NUM_LAYERS];

char letters[52] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                    'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                    'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z'};

void propagate_fwd(const matrix* weights, const f32* inputs, f32* results, const vector* biases) {
    sgemv_t_tuned(weights->data, inputs, results, weights->cols, weights->rows);
    // Add biases onto results
    vector_add_inplace(biases->len, biases->data, results);
}

// Minumum number of alligned_alloc without breaking things.
// This code f***ing sucks but its fast so uhhhh
u8 infer_reuse_layers_thread(vector* input, matrix** weights, vector** biases) {
    // Slightly larger than required for padding
    f32 out0[104] __attribute__((aligned(SIMD_ALIGN))) = {0};
    f32 out1[72] __attribute__((aligned(SIMD_ALIGN))) = {0};

    propagate_fwd(weights[0], input->data, out0, biases[0]);
    relu_inplace(out0, 98);

    propagate_fwd(weights[1], out0, out1, biases[1]);
    relu_inplace(out1, 65);

    memset(out0, 0, 50 * sizeof(f32));

    propagate_fwd(weights[2], out1, out0, biases[2]);
    relu_inplace(out0, 50);

    memset(out1, 0, 30 * sizeof(f32));

    propagate_fwd(weights[3], out0, out1, biases[3]);
    relu_inplace(out1, 30);

    memset(out0, 0, 25 * sizeof(f32));

    propagate_fwd(weights[4], out1, out0, biases[4]);
    relu_inplace(out0, 25);

    memset(out1, 0, 40 * sizeof(f32));

    propagate_fwd(weights[5], out0, out1, biases[5]);
    relu_inplace(out1, 40);

    memset(out0, 0, 52 * sizeof(f32));

    propagate_fwd(weights[6], out1, out0, biases[6]);
    softmax_inplace(out0, 52);

    u8 prediction = argmax(out0, 52);

    return prediction;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Not enough arguments. Usage: speed_cpu <path_to_model.txt> <tensors_dir/> <number_of_inferences>\n");
        return EXIT_FAILURE;
    }

    // Start timing
    struct timeval stop, start, preinf;
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

    read_model(weights, biases, argv[1]);

    // Transpose weights to column major
    for (int i = 0; i < NUM_LAYERS; i++)
        transpose_mat_inplace(weights[i]);

    // Set up preliminary counts and data
    const char* directory_path = argv[2];
    int input_count = file_count(directory_path);
    int iter_per_in = atoi(argv[3]);
    printf("Number of input tensors: %d\n", input_count);
    printf("Iterations per input: %d\n", iter_per_in);

    f32* tensors = (f32*)aligned_alloc(SIMD_ALIGN, TSIZE_ALIGN_BYTES * input_count);

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
            read_tensor((f32*)&tensors[TSIZE_ALIGN_BYTES / sizeof(f32) * (file_num - 1)], file_path);
        }
    }
    closedir(dir);
    free(file_path);
    free(file_num_str);

    // Time pre-inference
    gettimeofday(&preinf, NULL);
    printf("Pre inference (model read, tensor read, transpose) took %lu us\n",
           (preinf.tv_sec - start.tv_sec) * 1000000 + preinf.tv_usec - start.tv_usec);

    // int NUM_THREADS = sysconf(_SC_NPROCESSORS_ONLN);

    if (iter_per_in > 1)
#pragma omp parallel
    {
        int force = 0;
        u8* results_local = (u8*)malloc(input_count * sizeof(u8));

        for (int i = 0; i < input_count; i++) {
            // printf("Thread %d: Processing input %d\n", omp_get_thread_num(), i);

            vector* input = new_vec_aligned(TENSOR_SIZE);
            memcpy(input->data, (f32*)&tensors[TSIZE_ALIGN_BYTES / sizeof(f32) * i], TENSOR_SIZE * sizeof(f32));

#pragma omp for
            for (int j = 0; j < iter_per_in - 1; j++) {
                // Using global memory for model seems to be faster
                results_local[i] = infer_reuse_layers_thread(input, weights, biases);
                force += results_local[i];
            }

            free(input->data);
            free(input);
        }

        free(results_local);
        printf("Thread %2d: %d\n", omp_get_thread_num(), force);
    }

    // Output for csv
    vector* input = new_vec_aligned(TENSOR_SIZE);
    u8* results = (u8*)malloc(input_count * sizeof(u8));
    for (int i = 0; i < input_count; i++) {
        input->data = (f32*)&tensors[TSIZE_ALIGN_BYTES / sizeof(f32) * i];
        results[i] = infer_reuse_layers_thread(input, weights, biases);
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
    printf("Full run took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

    return EXIT_SUCCESS;
}
