cmake_minimum_required(VERSION 3.16)

# Set the project name
project(ichida-algo LANGUAGES C CXX CUDA)

# Set compiler flags
set(CMAKE_C_FLAGS "-O3 -march=native -ffast-math -funroll-loops -Wall -Wextra")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CUDA_FLAGS "-O3 -arch=sm_80")

set(SRC_DIR src)
set(INC_DIR include)
set(LIB_DIR lib)
set(CUDA_DIR ${SRC_DIR}/cuda)

# Source files
file(GLOB_RECURSE CPU_SOURCE_FILES ${SRC_DIR}/*.c)
file(GLOB_RECURSE CUDA_SOURCE_FILES ${CUDA_DIR}/*.cu)

# Exclude main_gpu.cu from CPU sources and main_cpu.c from GPU sources
list(REMOVE_ITEM CPU_SOURCE_FILES ${SRC_DIR}/main_gpu.cu)
list(REMOVE_ITEM CUDA_SOURCE_FILES ${SRC_DIR}/main_cpu.c)

include_directories(${INC_DIR})

# Create executables
add_executable(speed_cpu ${CPU_SOURCE_FILES})
add_executable(speed_gpu ${CUDA_SOURCE_FILES})

# Link libraries
target_link_libraries(speed_cpu m)
target_link_libraries(speed_gpu m)