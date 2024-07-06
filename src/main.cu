#include "matrix.cuh"
#include <dirent.h>
#include <mpi.h>
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
matrix** d_weights;
matrix** d_biases;

float* inputs;
float* d_inputs;
int* results;
int* d_results;

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

void read_tensor(float* a, const char* fileName) {
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
        a[i] = value;
        token = strtok(NULL, delimiter);
    }
    free(line);
    fclose(file);
}

__device__ void propagate_fwd(matrix* weights, float* input_layer, float* output_layer, matrix* biases) {
    matrix_mul(weights->data, input_layer, output_layer, weights->rows, weights->cols);
    matrix_add(output_layer, biases->data, biases->rows);
}

#define BLOCKS 108
#define THREADS_PER_BLOCK 1024

__global__ void infer(float* d_inputs, int* d_results, matrix** d_weights, matrix** d_biases, int it_per_input,
                      int in_num) {

    __shared__ float sharedInput[225];
    float out1[98];
    float out2[65];

    int num_threads = blockDim.x * gridDim.x;
    int thread_idx = (blockIdx.x * blockDim.x + threadIdx.x);

    float* input = (float*)&d_inputs[in_num * 225];

    if (threadIdx.x < 225) {
        sharedInput[threadIdx.x] = input[threadIdx.x];
    }
    __syncthreads();

    for (int i = thread_idx; i < it_per_input; i += num_threads) {
        propagate_fwd(d_weights[0], sharedInput, out1, d_biases[0]);
        relu(out1, 98);

        propagate_fwd(d_weights[1], out1, out2, d_biases[1]);
        relu(out2, 65);

        propagate_fwd(d_weights[2], out2, out1, d_biases[2]);
        relu(out1, 50);

        propagate_fwd(d_weights[3], out1, out2, d_biases[3]);
        relu(out2, 30);

        propagate_fwd(d_weights[4], out2, out1, d_biases[4]);
        relu(out1, 25);

        propagate_fwd(d_weights[5], out1, out2, d_biases[5]);
        relu(out2, 40);

        propagate_fwd(d_weights[6], out2, out1, d_biases[6]);
        softmax(out1, 52);

        d_results[in_num] = argmax(out1, 52);
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int TotalProcess, ProcessId;
    MPI_Comm_size(MPI_COMM_WORLD, &TotalProcess); // size
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcessId);    // gpuid

    if (argc < 4) {
        printf("Not enough arguments. Usage: speed_cpu <path_to_model.txt> <tensors_dir/> <number_of_inferences>\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // get no of gpu
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int deviceId = ProcessId % deviceCount;
    cudaSetDevice(deviceId);

    // Start timing
    struct timeval stop, start;
    gettimeofday(&start, NULL);

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

    CUDA_CHECK(cudaMalloc(&d_weights, NUM_LAYERS * sizeof(matrix*)));
    CUDA_CHECK(cudaMalloc(&d_biases, NUM_LAYERS * sizeof(matrix*)));
    for (int i = 0; i < NUM_LAYERS; i++) {
        matrix* a = copy_to_device(weights[i]);
        matrix* b = copy_to_device(biases[i]);
        CUDA_CHECK(cudaMemcpy(&(d_weights[i]), &a, sizeof(matrix*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&(d_biases[i]), &b, sizeof(matrix*), cudaMemcpyHostToDevice));
    }

    const char* directory_path = argv[2];
    struct dirent* entry;
    DIR* dir = opendir(directory_path);

    // Read and process inputs
    char* file_name = (char*)malloc((100) * sizeof(char));
    char* file_num_str = (char*)malloc((100) * sizeof(char));

    int file_num;
    int input_count = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            input_count++;
        }
    }
    results = (int*)malloc((input_count) * sizeof(int));
    inputs = (float*)malloc((input_count) * sizeof(float) * 225);

    cudaMalloc(&d_results, (input_count) * sizeof(int));
    cudaMalloc(&d_inputs, (input_count) * sizeof(float) * 225);

    dir = opendir(directory_path);
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            strcpy(file_num_str, entry->d_name);
            file_num_str[strlen(entry->d_name) - 7] = '\0';
            file_num = atoi(entry->d_name);
            strcpy(file_name, directory_path);
            strcat(file_name, "/");
            strcat(file_name, entry->d_name);
            read_tensor((float*)&inputs[(file_num - 1) * 225], file_name);
        }
    }

    free(file_name);
    free(file_num_str);
    closedir(dir);

    cudaMemcpy(d_inputs, inputs, sizeof(float) * 225 * input_count, cudaMemcpyHostToDevice);

    int it_num = atoi(argv[3]);
    // divide this doma       //when u launch 8 gpu it divide automatically yeah  //handles remainder
    int gpu_it_num = it_num / TotalProcess + (ProcessId < (it_num % TotalProcess) ? 1 : 0);

    struct timeval stop1, start1;
    gettimeofday(&start1, NULL);

    cudaDeviceSynchronize();
    for (int i = 0; i < input_count; i++) {
        infer<<<BLOCKS, THREADS_PER_BLOCK>>>(d_inputs, d_results, d_weights, d_biases, gpu_it_num, i);
        CUDA_CHECK(cudaGetLastError());
    }
    cudaDeviceSynchronize();

    cudaMemcpy(results, d_results, (input_count) * (sizeof(int)), cudaMemcpyDeviceToHost);
    gettimeofday(&stop1, NULL);
    printf("Process %d - Inference: %lu us\n", ProcessId,
           (stop1.tv_sec - start1.tv_sec) * 1000000 + stop1.tv_usec - start1.tv_usec);
    MPI_Finalize();
    // this cheat xd dan no verify xddd

    FILE* csv_file = fopen("results.csv", "w+");
    fprintf(csv_file, "image_number, guess\n");
    for (int i = 0; i < input_count; i++) {
        fprintf(csv_file, "%d, %c\n", i + 1, letters[results[i]]);
        printf("dan is gay =%d \n", ProcessId);
    }
    fclose(csv_file);

    // Time taken
    gettimeofday(&stop, NULL);
    printf("Process %d - Total: %lu us\n", ProcessId,
           (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

    return EXIT_SUCCESS;
}
