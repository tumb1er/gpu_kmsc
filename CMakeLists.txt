cmake_minimum_required(VERSION 3.5)
project(gpu_kmsc)

find_package(CUDA QUIET REQUIRED)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        utils/argparse.cpp
        utils/utils.cpp
        main.cpp
        )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../bin")

cuda_add_executable(gpu_kmsc ${SOURCE_FILES})
add_subdirectory(kmcuda/src)

include_directories(kmcuda/src)
link_directories(lib)

target_link_libraries(gpu_kmsc KMCUDA)