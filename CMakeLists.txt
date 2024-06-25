cmake_minimum_required(VERSION 3.16)

# Set the project name
project(ichida-algo)

set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra")
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

set(SRC_DIR src)
set(INC_DIR include)


# Source files
file(GLOB_RECURSE INCLUDE_FILES ${INC_DIR}/*.h)
file(GLOB_RECURSE SOURCE_FILES ${SRC_DIR}/*.c)

add_library(${PROJECT_NAME} SHARED ${INCLUDE_FILES} ${SOURCE_FILES})
include_directories(include)
add_executable(speed_cpu ${SOURCE_FILES})
target_link_libraries(speed_cpu m)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    PUBLIC_HEADER "${INCLUDE_FILES}"
)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)


