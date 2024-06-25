#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: speed_gpu <relative_path_to_weights_and_biases.txt> <relative_path_to_input_tensor_directory>"
    exit 1
fi

weights_and_biases=$1
input_tensor_dir=$2

binary="speed_gpu"

if [ ! -f "$binary" ]; then
    echo "Binary $binary not found!"
    exit 1
fi

start_time=$(date +%s)
./$binary "$weights_and_biases" "$input_tensor_dir"

end_time=$(date +%s)
execution_time=$((end_time - start_time))

if [ ! -f "results.csv" ]; then
    echo "Error: results.csv not found!"
    exit 1
else
    echo "results.csv found!"
fi

echo "Execution time: $execution_time seconds"
