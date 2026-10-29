#include <cstdlib>
#include <iostream>
#include <vector>


std::vector<float> runGelu(const std::vector<float>& input, int size);


int main() {
    int size = 10000000;

    std::vector<float> input(size);

    for (int i = 0; i < size; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX * 6.0f - 3.0f;
    }

    std::vector<float> output = runGelu(input, size);

    return 0;
}
