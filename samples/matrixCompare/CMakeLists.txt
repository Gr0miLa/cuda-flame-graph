add_executable(
    matrixCompare
    src/main.cpp
    src/cuda/matrixCompare.cu
)
set_property(TARGET matrixCompare PROPERTY CUDA_ARCHITECTURES
    "${PROFILER_SAMPLE_CUDA_ARCHITECTURES}")

target_include_directories(matrixCompare PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    src
)
target_link_libraries(matrixCompare PRIVATE
    CUDA::cudart
)
target_link_options(matrixCompare PRIVATE -rdynamic)
