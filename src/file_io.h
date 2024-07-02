#pragma once

#include "matrix.h"

int file_count(const char* dir_path);

void process_weights_str(matrix** weights, char* line, int layer);
void process_biases_str(vector** biases, char* line, int layer);

void read_model(matrix** weights, vector** biases, const char* file_name);
void read_tensor(f32* a, const char* file_name);