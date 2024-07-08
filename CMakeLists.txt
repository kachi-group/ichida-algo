cmake_minimum_required(VERSION 3.16)

project(ichida-algo LANGUAGES C CXX CUDA)

set(CMAKE_C_FLAGS "-O3 -march=native -ffast-math -funroll-loops -fopenmp -Wall -Wextra")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3 --use_fast_math -Xcompiler -march=native -unroll-aggressive -arch=sm_80")

set(INC_DIR include)
set(SRC_DIR src)
set(CUDA_SRC_DIR cuda/src)

include_directories(${INC_DIR})

file(GLOB_RECURSE SOURCE_FILES ${SRC_DIR}/*.c)

add_executable(speed_cpu ${SOURCE_FILES})
target_link_libraries(speed_cpu m pthread gomp)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

find_package(CUDA)

if(CUDA_FOUND)
    file(GLOB_RECURSE CUDA_SOURCE_FILES ${CUDA_SRC_DIR}/*.cu)
    add_executable(speed_gpu ${CUDA_SOURCE_FILES})
    set_target_properties(speed_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(speed_gpu m ${MPI_LIBRARIES})
else()
    message(STATUS "CUDA not found, only CPU version will be built.")
endif()


