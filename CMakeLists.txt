cmake_minimum_required(VERSION 3.10)

# Set the project name
project(ichida-algo)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

add_executable(speed_cpu src/main.c)