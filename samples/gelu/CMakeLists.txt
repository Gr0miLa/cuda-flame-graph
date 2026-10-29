add_executable(
    gelu
    src/main.cpp
    src/cuda/gelu.cu
)
set_property(TARGET gelu PROPERTY CUDA_ARCHITECTURES
    "${PROFILER_SAMPLE_CUDA_ARCHITECTURES}")

target_include_directories(gelu PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    src
)
target_link_libraries(gelu PRIVATE
    CUDA::cudart
)
target_link_options(gelu PRIVATE -rdynamic)
