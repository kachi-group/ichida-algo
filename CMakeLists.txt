cmake_minimum_required(VERSION 3.16)

# Set the project name
project(ichida-algo LANGUAGES C CXX CUDA)

# Set compiler flags
set(CMAKE_C_FLAGS "-O3 -march=native -ffast-math -funroll-loops -Wall -Wextra")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Ensure CUDA NVCC flags are set properly
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3 --use_fast_math -Xcompiler -march=native -unroll-aggressive -arch=sm_80")

set(INC_DIR include)
set(SRC_DIR src)

include_directories(${INC_DIR})

# Source files for GPU
file(GLOB_RECURSE CUDA_SOURCE_FILES ${SRC_DIR}/*.cu)

# Create GPU executable
add_executable(speed_gpu ${CUDA_SOURCE_FILES})
set_target_properties(speed_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(speed_gpu m)
