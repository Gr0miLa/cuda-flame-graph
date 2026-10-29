add_executable(
    softmax
    src/main.cpp
    src/cuda/softmax.cu
)
set_property(TARGET softmax PROPERTY CUDA_ARCHITECTURES
    "${PROFILER_SAMPLE_CUDA_ARCHITECTURES}")

target_include_directories(softmax PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    src
)
target_link_libraries(softmax PRIVATE
    CUDA::cudart
)
target_link_options(softmax PRIVATE -rdynamic)
