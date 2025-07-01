#include "NeuralNetwork.hpp"
#include <iostream>

int main() {
    // Создаём сеть с 3 входами, одним скрытым слоем из 5 нейронов и 2 выходами
    NeuralNetwork net({3, 5, 2});

    std::vector<float> input = {1.0f, 0.5f, -1.2f};
    std::vector<float> output = net.forward(input);

    for (float v : output) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    return 0;
}
