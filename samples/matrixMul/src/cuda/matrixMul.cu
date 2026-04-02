#include <vector>
#include <cstdlib>

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

std::vector<float> runMatrixMul(const std::vector<float>& a,
                                 const std::vector<float>& b, int size) {
  
  std::vector<float> c(size * size);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;

  cudaMalloc(&d_a, static_cast<unsigned long long>(size) * size * sizeof(float));
  cudaMalloc(&d_b, static_cast<unsigned long long>(size) * size * sizeof(float));
  cudaMalloc(&d_c, static_cast<unsigned long long>(size) * size * sizeof(float));

  cudaMemcpy(d_a, a.data(),
             static_cast<unsigned long long>(size) * size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(),
             static_cast<unsigned long long>(size) * size * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((size + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (size + threadsPerBlock.y - 1) / threadsPerBlock.y);

  MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, size);

  cudaMemcpy(c.data(), d_c,
             static_cast<unsigned long long>(size) * size * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return c;
}