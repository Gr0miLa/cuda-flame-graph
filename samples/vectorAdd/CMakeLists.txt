add_executable(
    vectorAdd
    src/main.cpp
    src/cuda/vectorAdd.cu
)

target_include_directories(vectorAdd PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    src
)
target_link_libraries(vectorAdd PRIVATE 
    CUDA::cudart
    # CUDA::cupti
    # profilerCupti
)
target_link_options(vectorAdd PRIVATE -rdynamic)
