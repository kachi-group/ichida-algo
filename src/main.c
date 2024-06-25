#include "matrix.h"
#include "util.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

matrix* weight[7];
matrix* biase[7];
int results[1000000];

void processWeight(char* line, int layer) {
    char* token;
    float value;
    const char* delimiter = ",";

    token = strtok(line, delimiter);
    for (int i = 0; i < weight[layer]->rows; i++) {
        for (int j = 0; j < weight[layer]->cols; j++) {
            value = strtof(token, NULL);
            (weight[layer]->data)[i][j] = value;
            token = strtok(NULL, delimiter);
        }
    }
}

void processBiase(char* line, int layer) {
    char* token;
    float value;
    const char* delimiter = ",";

    token = strtok(line, delimiter);

    for (int i = 0; i < biase[layer]->rows; i++) {
        value = strtof(token, NULL);
        (biase[layer]->data)[i][0] = value;
        token = strtok(NULL, delimiter);
    }
}

void readModel(const char* fileName) {
    FILE* file = fopen(fileName, "r");

    char* line = NULL;
    size_t len = 0;
    int line_number = 0;
    int layer = 0;

    while ((getline(&line, &len, file)) != -1) {
        if ((line_number - 1) % 4 == 0) {
            processWeight(line, layer);
        } else if ((line_number - 3) % 4 == 0) {
            processBiase(line, layer);
            layer++;
        }
        line_number++;
    }

    free(line);
    fclose(file);
}

void readInput(matrix* a, const char* fileName) {
    FILE* file = fopen(fileName, "r");
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    int line_number = 0;

    getline(&line, &len, file);
    char* token;
    float value;
    const char* delimiter = ",";
    token = strtok(line, delimiter);
    int size = 0;
    for (int i = 0; i < 225; i++) {
        value = strtof(token, NULL);
        (a->data)[i][0] = value;
        token = strtok(NULL, delimiter);
    }
    free(line);
    fclose(file);
}

void propagateForward(const matrix* weight, const matrix* input, matrix* nextLayer, const matrix* biase) {
    multiplyMatrices(weight, input, nextLayer);
    addMatrix(nextLayer, biase);
}

// Get result from output layer
int getResult(matrix* a) {
    int idx = 0;
    float res = (a->data)[0][0];
    for (int i = 0; i < a->rows; i++) {
        if (res < (a->data)[i][0]) {
            res = (a->data)[i][0];
            idx = i;
        }
    }
    return idx;
}

int inference(matrix* input) {
    matrix* layer[7];
    layer[0] = createMatrix(98, 1);
    layer[1] = createMatrix(65, 1);
    layer[2] = createMatrix(50, 1);
    layer[3] = createMatrix(30, 1);
    layer[4] = createMatrix(25, 1);
    layer[5] = createMatrix(40, 1);
    layer[6] = createMatrix(52, 1);

    propagateForward(weight[0], input, layer[0], biase[0]);
    ReLU(layer[0]);
    propagateForward(weight[1], layer[0], layer[1], biase[1]);
    ReLU(layer[1]);
    propagateForward(weight[2], layer[1], layer[2], biase[2]);
    ReLU(layer[2]);
    propagateForward(weight[3], layer[2], layer[3], biase[3]);
    ReLU(layer[3]);
    propagateForward(weight[4], layer[3], layer[4], biase[4]);
    ReLU(layer[4]);
    propagateForward(weight[5], layer[4], layer[5], biase[5]);
    ReLU(layer[5]);

    propagateForward(weight[6], layer[5], layer[6], biase[6]);
    softmax(layer[6]);

    return getResult(layer[6]);
}

int main(int argc, char* argv[]) {
    // Load model (The memory of those code should be initialize during compile time to enchance the speed)
    weight[0] = createMatrix(98, 225);
    for (int i = 0; i < weight[0]->rows; i++) {
        for (int j = 0; j < weight[0]->cols; j++) {
            (weight[0]->data)[i][j] = w0[i][j];
        }
    }

    weight[1] = createMatrix(65, 98);
    for (int i = 0; i < weight[1]->rows; i++) {
        for (int j = 0; j < weight[1]->cols; j++) {
            (weight[1]->data)[i][j] = w1[i][j];
        }
    }
    weight[2] = createMatrix(50, 65);
    for (int i = 0; i < weight[2]->rows; i++) {
        for (int j = 0; j < weight[2]->cols; j++) {
            (weight[2]->data)[i][j] = w2[i][j];
        }
    }
    weight[3] = createMatrix(30, 50);
    for (int i = 0; i < weight[3]->rows; i++) {
        for (int j = 0; j < weight[3]->cols; j++) {
            (weight[3]->data)[i][j] = w3[i][j];
        }
    }
    weight[4] = createMatrix(25, 30);
    for (int i = 0; i < weight[4]->rows; i++) {
        for (int j = 0; j < weight[4]->cols; j++) {
            (weight[4]->data)[i][j] = w4[i][j];
        }
    }
    weight[5] = createMatrix(40, 25);
    for (int i = 0; i < weight[5]->rows; i++) {
        for (int j = 0; j < weight[5]->cols; j++) {
            (weight[5]->data)[i][j] = w5[i][j];
        }
    }
    weight[6] = createMatrix(52, 40);
    for (int i = 0; i < weight[6]->rows; i++) {
        for (int j = 0; j < weight[6]->cols; j++) {
            (weight[6]->data)[i][j] = w6[i][j];
        }
    }

    biase[0] = createMatrix(98, 1);
    for (int i = 0; i < biase[0]->rows; i++) {
        (biase[0]->data)[i][0] = b0[i][0];
    }

    biase[1] = createMatrix(65, 1);
    for (int i = 0; i < biase[1]->rows; i++) {
        (biase[1]->data)[i][0] = b1[i][0];
    }

    biase[2] = createMatrix(50, 1);
    for (int i = 0; i < biase[2]->rows; i++) {
        (biase[2]->data)[i][0] = b2[i][0];
    }

    biase[3] = createMatrix(30, 1);
    for (int i = 0; i < biase[3]->rows; i++) {
        (biase[3]->data)[i][0] = b3[i][0];
    }

    biase[4] = createMatrix(25, 1);
    for (int i = 0; i < biase[4]->rows; i++) {
        (biase[4]->data)[i][0] = b4[i][0];
    }

    biase[5] = createMatrix(40, 1);
    for (int i = 0; i < biase[5]->rows; i++) {
        (biase[5]->data)[i][0] = b5[i][0];
    }

    biase[6] = createMatrix(52, 1);
    for (int i = 0; i < biase[6]->rows; i++) {
        (biase[6]->data)[i][0] = b6[i][0];
    }

    // Run program
    const char* directory_path = argv[2];
    struct dirent* entry;
    DIR* dir = opendir(directory_path);

    matrix* input = createMatrix(225, 1);

    // Read and process inputs
    char* fileName = (char*)malloc((100) * sizeof(char));
    char* fileNumStr = (char*)malloc((100) * sizeof(char));

    int fileNum;
    int size = 0;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            size++;
            strcpy(fileNumStr, entry->d_name);
            fileNumStr[strlen(entry->d_name) - 7] = '\0';
            fileNum = atoi(entry->d_name);
            strcpy(fileName, directory_path);
            strcat(fileName, "/");
            strcat(fileName, entry->d_name);
            readInput(input, fileName);
            results[fileNum] = inference(input);
        }
    }
    free(fileName);
    free(fileNumStr);
    closedir(dir);

    // Write to csv file
    FILE* fpt;
    fpt = fopen("results.csv", "w+");
    fprintf(fpt, "image_number, guess\n");
    for (int i = 1; i <= size; i++) {
        fprintf(fpt, "%d, %c\n", i, letters[results[i]]);
    }

    return EXIT_SUCCESS;
}