#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>


__global__ void MatrixMulKernel(const float* a, const float* b, float* c,
                                int size) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (col < size && row < size) {
    float sum = 0.0f;
    for (int k = 0; k < size; ++k) {
      sum += a[row * size + k] * b[k * size + col];
    }
    c[row * size + col] = sum;
  }
}

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        std::cerr << operation << ": "
                  << cudaGetErrorString(result) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    int size = 4096;
    int repeats = 5;

    unsigned long long elements =
        static_cast<unsigned long long>(size) * size;
    unsigned long long bytes = elements * sizeof(float);

    std::vector<float> a(elements);
    std::vector<float> b(elements);
    std::vector<float> c(elements);

    for (unsigned long long i = 0; i < elements; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    checkCuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    checkCuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");
    checkCuda(cudaMalloc(&d_c, bytes), "cudaMalloc d_c");

    checkCuda(
        cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice),
        "copy a to GPU"
    );

    checkCuda(
        cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice),
        "copy b to GPU"
    );

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (size + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (size + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, size);

    checkCuda(cudaGetLastError(), "warm-up kernel launch");
    checkCuda(cudaDeviceSynchronize(), "warm-up synchronization");

    cudaEvent_t start;
    cudaEvent_t stop;

    checkCuda(cudaEventCreate(&start), "create start event");
    checkCuda(cudaEventCreate(&stop), "create stop event");

    checkCuda(cudaEventRecord(start), "record start event");

    for (int i = 0; i < repeats; ++i) {
        MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, size);
    }

    checkCuda(cudaGetLastError(), "measured kernel launch");
    checkCuda(cudaEventRecord(stop), "record stop event");
    checkCuda(cudaEventSynchronize(stop), "wait for stop event");

    float totalMilliseconds = 0.0f;

    checkCuda(
        cudaEventElapsedTime(&totalMilliseconds, start, stop),
        "calculate elapsed time"
    );

    float averageMilliseconds = totalMilliseconds / repeats;

    std::cerr << "MatrixMulKernel"
              << " size=" << size
              << " repeats=" << repeats
              << " total_ms=" << totalMilliseconds
              << " average_ms=" << averageMilliseconds
              << '\n';
    checkCuda(
        cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost),
        "copy result to CPU"
    );

    float expected = 0.0f;

    for (int k = 0; k < size; ++k) {
        expected += a[k] * b[k * size];
    }

    float difference = std::abs(c[0] - expected);
    float tolerance = std::abs(expected) * 0.001f + 0.001f;

    bool correct = difference <= tolerance;

    if (!correct) {
        std::cerr << "Incorrect result:"
                << " expected=" << expected
                << " actual=" << c[0]
                << '\n';
    }

    checkCuda(cudaEventDestroy(start), "destroy start event");
    checkCuda(cudaEventDestroy(stop), "destroy stop event");

    checkCuda(cudaFree(d_a), "free d_a");
    checkCuda(cudaFree(d_b), "free d_b");
    checkCuda(cudaFree(d_c), "free d_c");

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}