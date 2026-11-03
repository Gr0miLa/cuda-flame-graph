add_executable(
    matrixMulOverhead
    matrixMul.cu
)

set_property(
    TARGET matrixMulOverhead
    PROPERTY CUDA_ARCHITECTURES "${PROFILER_SAMPLE_CUDA_ARCHITECTURES}"
)

target_link_libraries(
    matrixMulOverhead
    PRIVATE CUDA::cudart
)

target_link_options(
    matrixMulOverhead
    PRIVATE -rdynamic
)