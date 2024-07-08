#!/usr/bin/env bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <relative_path_to_weights_and_biases.txt> <relative_path_to_input_tensor_directory> <inferences>"
    exit 1
fi

weights_and_biases=$1
input_tensor_dir=$2
inferences=$3

binary="speed_gpu"

if [ ! -f "$binary" ]; then
    echo "Binary $binary not found!"
    exit 1
fi

start_time=$(date +%s%3N)
./$binary "$weights_and_biases" "$input_tensor_dir" "$inferences"

end_time=$(date +%s%3N)
execution_time=$((end_time - start_time))

if [ ! -f "results.csv" ]; then
    echo "[ERROR] results.csv not found!"
    exit 1
else
    echo "[SUCCESS] results.csv found!"
fi

echo "Execution time: $execution_time ms"
