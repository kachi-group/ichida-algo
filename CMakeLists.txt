cmake_minimum_required(VERSION 3.16)

project(ichida-algo LANGUAGES C)

set(CMAKE_C_FLAGS "-O3 -march=native -ffast-math -funroll-loops -fopenmp -Wall -Wextra -Wpedantic")
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

set(SOURCE_DIR src)
set(CUDA_SRC_DIR cudasrc)

file(GLOB SOURCE_FILES ${SOURCE_DIR}/*.c)

add_executable(speed_cpu ${SOURCE_FILES})
target_link_libraries(speed_cpu m pthread gomp)

find_package(CUDA)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "80")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -O3 --use_fast_math -Xcompiler -march=native -unroll-aggressive -arch=sm_80")
    file(GLOB_RECURSE CUDA_SOURCE_FILES ${CUDA_SRC_DIR}/*.cu)
    add_executable(speed_gpu ${CUDA_SOURCE_FILES})
    set_target_properties(speed_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if(COMPILE_MPI)
        find_package(MPI)
        if(MPI_FOUND)
            include_directories(${MPI_INCLUDE_PATH})
            target_link_libraries(speed_gpu m ${MPI_LIBRARIES})
            set_property(TARGET speed_gpu PROPERTY COMPILE_DEFINITIONS USE_MPI)
        else()
            message(STATUS "MPI not found. Please install library to compile with MPI enabled.")
        endif(MPI_FOUND)
        
    else() 
        target_link_libraries(speed_gpu m)
    endif()

else()
    message(STATUS "CUDA not found, only CPU version will be built.")
endif()


