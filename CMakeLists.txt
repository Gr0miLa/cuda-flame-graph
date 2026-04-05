cmake_minimum_required(VERSION 3.18)
project(CUPTI_Stages LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_profiler SHARED 
    src/profiler.cpp 
    src/interface.cpp
)
target_include_directories(cuda_profiler PRIVATE include)
target_link_libraries(cuda_profiler PRIVATE CUDA::cudart CUDA::cupti dl)

set_target_properties(cuda_profiler PROPERTIES OUTPUT_NAME "cupti_prof")

add_executable(pti_loader loader/loader.cc)

target_compile_definitions(pti_loader PRIVATE TOOL_NAME=cupti_prof)

target_include_directories(pti_loader PRIVATE loader utils)

target_link_libraries(pti_loader PRIVATE pthread dl)

add_subdirectory(samples)