#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int rows;
    int cols;
    float** data;
} matrix;

matrix* weight[7];
matrix* biase[7];

matrix* createMatrix(int rows, int cols) {
    matrix* res = (matrix*)malloc(sizeof(matrix));
    res->rows = rows;
    res->cols = cols;
    res->data = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        res->data[i] = (float*)malloc(cols * sizeof(float));
    }
    return res;
}

void multiplyMatrices(const matrix* a, const matrix* b, const matrix* result) {
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            float sum = 0;
            for (int k = 0; k < a->cols; k++) {
                sum += (a->data)[i][k] * ((b->data)[k][j]);
            }
            (result->data)[i][j] = sum;
        }
    }
}

void addMatrix(matrix* a, const matrix* b) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            (a->data)[i][j] += (b->data)[i][j];
        }
    }
}

void ReLU(matrix* a) {
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            if ((a->data)[i][j] < (float)0)
                (a->data)[i][j] = (float)0;
        }
    }
}

void softmax(matrix* a) {
    float res = (float)0;
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            res += exp((a->data)[i][j]);
        }
    }
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            (a->data)[i][j] /= res;
        }
    }
}

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

void readModel() {
    FILE* file = fopen("../weights_and_biases.txt", "r");

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

void readInput(matrix* a, char* fileName) {
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

int getResult(matrix* a) {
    int idx = 0;
    float res = INT32_MIN;
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

int main() {

    // Load model
    weight[0] = createMatrix(98, 225);
    weight[1] = createMatrix(65, 98);
    weight[2] = createMatrix(50, 65);
    weight[3] = createMatrix(30, 50);
    weight[4] = createMatrix(25, 30);
    weight[5] = createMatrix(40, 25);
    weight[6] = createMatrix(52, 40);

    biase[0] = createMatrix(98, 1);
    biase[1] = createMatrix(65, 1);
    biase[2] = createMatrix(50, 1);
    biase[3] = createMatrix(30, 1);
    biase[4] = createMatrix(25, 1);
    biase[5] = createMatrix(40, 1);
    biase[6] = createMatrix(52, 1);

    readModel();

    char str[10];
    char fileName[100];
    matrix* input = createMatrix(255, 1);

    for (int i = 1; i <= 52; i++) {
        if (i < 10) {
            sprintf(str, "0%d", i);
        } else {
            sprintf(str, "%d", i);
        }
        strcpy(fileName, "../tensors/");
        strcat(fileName, str);
        strcat(fileName, "out.txt");

        // Read txt file
        readInput(input, fileName);

        // Check result
        if (inference(input) + 1 == i) {
            printf("Test %d correct ✅\n", i);
        } else {
            printf("Test %d incorrect ❌\n", i);
        }
    }
}