#include <cstdlib>
#include <iostream>
#include <vector>


std::vector<float> runRelu(const std::vector<float>& input, int size);


int main() {
    int size = 10000000;

    std::vector<float> input(size);

    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }

    std::vector<float> output = runRelu(input, size);

    return 0;
}
