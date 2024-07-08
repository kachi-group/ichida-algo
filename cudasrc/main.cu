#include "matrix.cuh"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

#define NUM_LAYERS 7
#define TENSOR_LENGTH 225

#define BLOCKS 108
#define THREADS_PER_BLOCK 1024

matrix* weights[NUM_LAYERS];
matrix* biases[NUM_LAYERS];
f32* inputs;
int* results;

// Device memory
matrix** d_weights;
matrix** d_biases;
f32* d_inputs;
int* d_results;

char letters[52] = {'A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'H', 'h', 'I', 'i',
                    'J', 'j', 'K', 'k', 'L', 'l', 'M', 'm', 'N', 'n', 'O', 'o', 'P', 'p', 'Q', 'q', 'R', 'r',
                    'S', 's', 'T', 't', 'U', 'u', 'V', 'v', 'W', 'w', 'X', 'x', 'Y', 'y', 'Z', 'z'};

void process_weights_str(char* line, int layer) {
    char* token;
    f32 value;
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
    f32 value;
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

void read_tensor(f32* out, const char* fileName) {
    FILE* file = fopen(fileName, "r");
    char* line = NULL;
    size_t len = 0;

    if (getline(&line, &len, file) == -1) {
        perror("Could not read tensor file.\n");
        exit(EXIT_FAILURE);
    }

    char* token;
    f32 value;
    const char* delimiter = ",";
    token = strtok(line, delimiter);

    for (int i = 0; i < 225; i++) {
        value = strtof(token, NULL);
        out[i] = value;
        token = strtok(NULL, delimiter);
    }
    free(line);
    fclose(file);
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

__device__ void propagate_fwd(matrix* weights, f32* input_layer, f32* output_layer, matrix* biases) {
    matrix_mul(weights->data, input_layer, output_layer, weights->rows, weights->cols);
    matrix_add(output_layer, biases->data, biases->rows);
}

__global__ void infer(f32* d_inputs, int* d_results, matrix** d_weights, matrix** d_biases, int it_per_input,
                      int in_num) {

    __shared__ f32 shared_input[TENSOR_LENGTH];
    f32 out1[98];
    f32 out2[65];

    int num_threads = blockDim.x * gridDim.x;
    int thread_idx = (blockIdx.x * blockDim.x + threadIdx.x);

    f32* input = (f32*)&d_inputs[in_num * TENSOR_LENGTH];

    if (threadIdx.x < TENSOR_LENGTH) {
        shared_input[threadIdx.x] = input[threadIdx.x];
    }
    __syncthreads();

    for (int i = thread_idx; i < it_per_input; i += num_threads) {
        propagate_fwd(d_weights[0], shared_input, out1, d_biases[0]);
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

    if (argc < 4) {
        printf("Not enough arguments. Usage: speed_cpu <path_to_model.txt> <tensors_dir/> <number_of_inferences>\n");
#ifdef USE_MPI
        MPI_Finalize();
#endif
        return EXIT_FAILURE;
    }

#ifdef USE_MPI
    // Initialise GPU environment
    MPI_Init(&argc, &argv);
    int num_proccesses, process_id;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proccesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

    int device_count;
    cudaGetDeviceCount(&device_count);
    int device_id = process_id % device_count;
    cudaSetDevice(device_id);
    printf("MPI device id: %d\n", device_id);
#endif

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

    // Copy model to GPU
    cudaMalloc(&d_weights, NUM_LAYERS * sizeof(matrix*));
    cudaMalloc(&d_biases, NUM_LAYERS * sizeof(matrix*));
    for (int i = 0; i < NUM_LAYERS; i++) {
        matrix* layer_weight = copy_to_device(weights[i]);
        matrix* layer_bias = copy_to_device(biases[i]);
        cudaMemcpy(&(d_weights[i]), &layer_weight, sizeof(matrix*), cudaMemcpyHostToDevice);
        cudaMemcpy(&(d_biases[i]), &layer_bias, sizeof(matrix*), cudaMemcpyHostToDevice);
    }

    const char* directory_path = argv[2];
    int input_count = file_count(directory_path);
    int num_its = atoi(argv[3]);

    results = (int*)malloc((input_count) * sizeof(int));
    inputs = (f32*)malloc((input_count) * sizeof(f32) * TENSOR_LENGTH);
    cudaMalloc(&d_results, (input_count) * sizeof(int));
    cudaMalloc(&d_inputs, (input_count) * sizeof(f32) * TENSOR_LENGTH);

    // Read and process inputs
    char* file_name = (char*)malloc((100) * sizeof(char));
    char* file_num_str = (char*)malloc((100) * sizeof(char));

    struct dirent* entry;
    DIR* dir = opendir(directory_path);
    dir = opendir(directory_path);
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            strcpy(file_num_str, entry->d_name);
            file_num_str[strlen(entry->d_name) - 7] = '\0';
            int file_num = atoi(entry->d_name);
            strcpy(file_name, directory_path);
            strcat(file_name, "/");
            strcat(file_name, entry->d_name);
            read_tensor((f32*)&inputs[(file_num - 1) * 225], file_name);
        }
    }
    free(file_name);
    free(file_num_str);
    closedir(dir);

    // Move input array to GPU memory
    cudaMemcpy(d_inputs, inputs, sizeof(f32) * 225 * input_count, cudaMemcpyHostToDevice);

#ifdef USE_MPI
    int it_per_gpu = num_its / num_proccesses + (process_id < (num_its % num_proccesses) ? 1 : 0);
#else
    int it_per_gpu = num_its;
#endif

    struct timeval stop_inf, start_inf;
    gettimeofday(&start_inf, NULL);

    cudaDeviceSynchronize();
    for (int i = 0; i < input_count; i++) {
        infer<<<BLOCKS, THREADS_PER_BLOCK>>>(d_inputs, d_results, d_weights, d_biases, it_per_gpu, i);
    }
    cudaDeviceSynchronize();

#ifdef USE_MPI
    if (process_id == 0) {
#endif
        cudaMemcpy(results, d_results, (input_count) * (sizeof(int)), cudaMemcpyDeviceToHost);
        gettimeofday(&stop_inf, NULL);
#ifdef USE_MPI
        printf("Process %d - Inference: %lu us\n", process_id,
               (stop_inf.tv_sec - start_inf.tv_sec) * 1000000 + stop_inf.tv_usec - start_inf.tv_usec);
#endif

        // Print output to csv
        FILE* csv_file = fopen("results.csv", "w+");
        fprintf(csv_file, "image_number, guess\n");
        for (int i = 0; i < input_count; i++) {
            fprintf(csv_file, "%d, %c\n", i + 1, letters[results[i]]);
        }
        fclose(csv_file);
#ifdef USE_MPI
    }
#endif

    // Time taken
    gettimeofday(&stop, NULL);
    printf("Total: %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return EXIT_SUCCESS;
}
