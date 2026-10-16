cmake_minimum_required(VERSION 3.24)
project(CUPTI_Stages LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(PROFILER_SAMPLE_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures used to build profiler sample applications")

find_package(CUDAToolkit REQUIRED)

add_library(cuda_profiler SHARED 
    src/profiler.cpp 
    src/cpu_profiler.cpp
    src/gpu_profiler.cpp
    src/stack_resolver.cpp
    src/interface.cpp
)
target_include_directories(cuda_profiler PRIVATE include)
target_link_libraries(cuda_profiler PRIVATE CUDA::cudart CUDA::cupti CUDA::cuda_driver dl)

set_target_properties(cuda_profiler PROPERTIES OUTPUT_NAME "cupti_prof")

add_executable(cuprof loader/loader.cc)

target_compile_definitions(cuprof PRIVATE TOOL_NAME=cupti_prof)

target_include_directories(cuprof PRIVATE loader utils)

target_link_libraries(cuprof PRIVATE pthread dl)

add_subdirectory(samples)
