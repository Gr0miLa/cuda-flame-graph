#include <vector>

__global__ void ReLUKernel(const float* input, float* output, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

std::vector<float> runRelu(const std::vector<float>& input, int size) {
    std::vector<float> output(size);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, static_cast<unsigned long long>(size) * sizeof(float));
    cudaMalloc(&d_output, static_cast<unsigned long long>(size) * sizeof(float));

    cudaMemcpy(d_input, input.data(),
               static_cast<unsigned long long>(size) * sizeof(float),
               cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    ReLUKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, size);

    cudaMemcpy(output.data(), d_output,
               static_cast<unsigned long long>(size) * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
