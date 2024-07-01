cmake_minimum_required(VERSION 3.16)

# Set the project name
project(ichida-algo)

set(CMAKE_C_FLAGS "-O3 -march=native -ffast-math -funroll-loops -Wall -Wextra")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(SRC_DIR src)
set(INC_DIR include)
set(LIB_DIR lib)
set(TEST_DIR test)
set(BENCHMARK_DIR benchmark)

# Source files
file(GLOB_RECURSE SOURCE_FILES ${SRC_DIR}/*.c)

include_directories(include)

add_executable(speed_cpu ${SOURCE_FILES})
add_executable(benchmark ${SRC_DIR}/matrix.c ${BENCHMARK_DIR}/benchmark.c)

target_link_libraries(speed_cpu m)
target_link_libraries(benchmark m)



