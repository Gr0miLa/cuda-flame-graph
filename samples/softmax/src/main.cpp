#include <cstdlib>
#include <iostream>
#include <vector>


std::vector<float> runSoftmax(const std::vector<float>& input,
                              int rows, int columns);


int main() {
    int rows = 16384;
    int columns = 1024;

    std::vector<float> input(static_cast<unsigned long long>(rows) * columns);

    for (unsigned long long i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
    }

    std::vector<float> output = runSoftmax(input, rows, columns);

    return 0;
}
