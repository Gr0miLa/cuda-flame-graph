#include <float.h>
#include <vector>

__global__ void SoftmaxKernel(const float* input, float* output, int columns) {
    __shared__ float values[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int rowOffset = row * columns;

    float localMax = -FLT_MAX;
    for (int column = tid; column < columns; column += blockDim.x) {
        localMax = fmaxf(localMax, input[rowOffset + column]);
    }

    values[tid] = localMax;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            values[tid] = fmaxf(values[tid], values[tid + offset]);
        }
        __syncthreads();
    }

    float rowMax = values[0];
    float localSum = 0.0f;

    for (int column = tid; column < columns; column += blockDim.x) {
        localSum += expf(input[rowOffset + column] - rowMax);
    }

    values[tid] = localSum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            values[tid] += values[tid + offset];
        }
        __syncthreads();
    }

    float rowSum = values[0];
    for (int column = tid; column < columns; column += blockDim.x) {
        output[rowOffset + column] =
            expf(input[rowOffset + column] - rowMax) / rowSum;
    }
}

std::vector<float> runSoftmax(const std::vector<float>& input,
                              int rows, int columns) {
    unsigned long long size = static_cast<unsigned long long>(rows) * columns;
    std::vector<float> output(size);

    float* d_input = nullptr;
    float* d_output = nullptr;

    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));

    cudaMemcpy(d_input, input.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    SoftmaxKernel<<<rows, threadsPerBlock>>>(d_input, d_output, columns);

    cudaMemcpy(output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
