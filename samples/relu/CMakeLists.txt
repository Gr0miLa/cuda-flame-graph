add_executable(
    relu
    src/main.cpp
    src/cuda/relu.cu
)
set_property(TARGET relu PROPERTY CUDA_ARCHITECTURES
    "${PROFILER_SAMPLE_CUDA_ARCHITECTURES}")

target_include_directories(relu PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    src
)
target_link_libraries(relu PRIVATE
    CUDA::cudart
)
target_link_options(relu PRIVATE -rdynamic)
