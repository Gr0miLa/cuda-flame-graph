#include <iostream>
#include <cstdlib>
#include <vector>


std::vector<float> runMatrixMul(const std::vector<float>& a,
                                const std::vector<float>& b, int size);

int main() {
    int size = 2048;

    std::vector<float> A(size * size);
    std::vector<float> B(size * size);

    for (int i = 0; i < size * size; ++i) {
        A[i] = rand();
        B[i] = rand();
    }

    std::vector<float> C = runMatrixMul(A, B, size);

    return 0;
}
