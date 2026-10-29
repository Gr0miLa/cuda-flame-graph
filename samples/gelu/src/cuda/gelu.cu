#include <vector>

__global__ void GELUKernel(const float* input, float* output, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = 0.7978845608f * (x + 0.044715f * x3);
        output[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

std::vector<float> runGelu(const std::vector<float>& input, int size) {
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

    GELUKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, size);

    cudaMemcpy(output.data(), d_output,
               static_cast<unsigned long long>(size) * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
