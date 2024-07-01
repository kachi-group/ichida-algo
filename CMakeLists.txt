cmake_minimum_required(VERSION 3.16)

# Set the project name
project(ichida-algo LANGUAGES C CXX CUDA)

# Set compiler flags
set(CMAKE_C_FLAGS "-O3 -march=native -ffast-math -funroll-loops -Wall -Wextra")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_VERBOSE_MAKEFILE ON)

# Ensure CUDA NVCC flags are set properly
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_75")

set(SRC_DIR src)
set(INC_DIR include)
set(CUDA_DIR cuda/src)

include_directories(${INC_DIR})

# Option to select only CPU or GPU build
option(BUILD_CPU "Build CPU code" ON)
option(BUILD_GPU "Build GPU code" OFF)

if (BUILD_CPU)
    # Source files for CPU
    file(GLOB_RECURSE CPU_SOURCE_FILES ${SRC_DIR}/*.c)

    # Create CPU executable
    add_executable(speed_cpu ${CPU_SOURCE_FILES})
    target_link_libraries(speed_cpu m)
endif()

if (BUILD_GPU)
    # Source files for GPU
    file(GLOB_RECURSE CUDA_SOURCE_FILES ${CUDA_DIR}/*.cu)

    # Create GPU executable
    add_executable(speed_gpu ${CUDA_SOURCE_FILES})
    set_target_properties(speed_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(speed_gpu m)
endif()