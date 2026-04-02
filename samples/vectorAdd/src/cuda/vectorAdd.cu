#include <vector>

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements) {
        C[i] = A[i] + B[i] + 0.0f;
    }
}

std::vector<float> runVectorAdd(const std::vector<float>& a,
                                 const std::vector<float>& b, int size) {
    std::vector<float> c(size);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaMalloc(&d_a, static_cast<unsigned long long>(size) * sizeof(float));
    cudaMalloc(&d_b, static_cast<unsigned long long>(size) * sizeof(float));
    cudaMalloc(&d_c, static_cast<unsigned long long>(size) * sizeof(float));

    cudaMemcpy(d_a, a.data(),
                static_cast<unsigned long long>(size) * sizeof(float),
                cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(),
                static_cast<unsigned long long>(size) * sizeof(float),
                cudaMemcpyHostToDevice);
    
    auto threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    vectorAddKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c.data(), d_c,
              static_cast<unsigned long long>(size) * sizeof(float),
              cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
