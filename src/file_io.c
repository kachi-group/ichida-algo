#include "matrix.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void process_weights_str(matrix** weights, char* line, int layer) {
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

void process_biases_str(vector** biases, char* line, int layer) {
    char* token;
    float value;
    const char* delimiter = ",";

    token = strtok(line, delimiter);

    int n = biases[layer]->len;
    for (int i = 0; i < n; i++) {
        value = strtof(token, NULL);
        (biases[layer]->data)[i] = value;
        token = strtok(NULL, delimiter);
    }
}

void read_model(matrix** weights, vector** biases, const char* file_name) {
    FILE* file = fopen(file_name, "r");

    char* line = NULL;
    size_t len = 0;
    int line_number = 0;
    int layer = 0;

    while ((getline(&line, &len, file)) != -1) {
        if ((line_number - 1) % 4 == 0) {
            process_weights_str(weights, line, layer);
        } else if ((line_number - 3) % 4 == 0) {
            process_biases_str(biases, line, layer);
            layer++;
        }
        line_number++;
    }

    free(line);
    fclose(file);
}

void read_tensor(f32* a, const char* file_name) {
    FILE* file = fopen(file_name, "r");
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