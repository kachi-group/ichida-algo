cmake_minimum_required(VERSION 3.10)

# Set the project name
project(hpc)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

add_executable(model src/main.c)