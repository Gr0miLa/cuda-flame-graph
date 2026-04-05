#include <iostream>
#include <cstdlib>
#include <vector>


std::vector<float> runVectorAdd(const std::vector<float>& a,
                                const std::vector<float>& b, int size);


int main() {
    int size = 10000000;

    std::vector<float> A(size);
    std::vector<float> B(size);

    for (int i = 0; i < size; ++i) {
        A[i] = rand();
        B[i] = rand();
    }

    std::vector<float> C = runVectorAdd(A, B, size);

    return 0;
}
