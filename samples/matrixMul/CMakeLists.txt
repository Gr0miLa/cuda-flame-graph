add_executable(
    matrixMul
    src/main.cpp
    src/cuda/matrixMul.cu
)
set_property(TARGET matrixMul PROPERTY CUDA_ARCHITECTURES
    "${PROFILER_SAMPLE_CUDA_ARCHITECTURES}")

target_include_directories(matrixMul PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    src
)
target_link_libraries(matrixMul PRIVATE 
    CUDA::cudart
    # CUDA::cupti
    # profilerCupti
)
target_link_options(matrixMul PRIVATE -rdynamic)
