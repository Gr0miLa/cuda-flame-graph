#include <vector>

__global__ void BasicKernel(const float* a, const float* b,
                            volatile float* c, int size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < size && row < size) {
        int index = row * size + col;
        c[index] = 0.0f;

        for (int k = 0; k < size; ++k) {
            c[index] = c[index] + a[row * size + k] * b[k * size + col];
        }
    }
}

__global__ void RegisterKernel(const float* a, const float* b,
                               float* c, int size) {
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

__global__ void TiledKernel(const float* a, const float* b,
                            float* c, int size) {
    __shared__ float tileA[16][16];
    __shared__ float tileB[16][16];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;

    for (int tile = 0; tile < (size + 15) / 16; ++tile) {
        int aCol = tile * 16 + threadIdx.x;
        int bRow = tile * 16 + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            row < size && aCol < size ? a[row * size + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            bRow < size && col < size ? b[bRow * size + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < 16; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (col < size && row < size) {
        c[row * size + col] = sum;
    }
}

std::vector<float> runBasic(const std::vector<float>& a,
                            const std::vector<float>& b, int size) {
    std::vector<float> c(size * size);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    unsigned long long bytes =
        static_cast<unsigned long long>(size) * size * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((size + 15) / 16, (size + 15) / 16);

    BasicKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}

std::vector<float> runRegister(const std::vector<float>& a,
                               const std::vector<float>& b, int size) {
    std::vector<float> c(size * size);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    unsigned long long bytes =
        static_cast<unsigned long long>(size) * size * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((size + 15) / 16, (size + 15) / 16);

    RegisterKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}

std::vector<float> runTiled(const std::vector<float>& a,
                            const std::vector<float>& b, int size) {
    std::vector<float> c(size * size);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    unsigned long long bytes =
        static_cast<unsigned long long>(size) * size * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((size + 15) / 16, (size + 15) / 16);

    TiledKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, size);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}
