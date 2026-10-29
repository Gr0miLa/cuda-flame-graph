#include <cstdlib>
#include <iostream>
#include <vector>


std::vector<float> runBasic(const std::vector<float>& a,
                            const std::vector<float>& b, int size);
std::vector<float> runRegister(const std::vector<float>& a,
                               const std::vector<float>& b, int size);
std::vector<float> runTiled(const std::vector<float>& a,
                            const std::vector<float>& b, int size);


int main() {
    int size = 1024;

    std::vector<float> A(size * size);
    std::vector<float> B(size * size);

    for (int i = 0; i < size * size; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    std::vector<float> basic = runBasic(A, B, size);
    std::vector<float> registerResult = runRegister(A, B, size);
    std::vector<float> tiled = runTiled(A, B, size);

    return 0;
}
