#include <iostream>
#include "BersbotsAILib.hpp"

int main() {
    AIbot layer(3, 2); // 3 входа, 2 выхода
    layer.setUseSigmoid(true);

    std::vector<float> input = {0.5, -0.3, 0.8};
    std::vector<float> output = layer.forward(input);

    std::cout << "Output:\n";
    for (float y : output) {
        std::cout << y << " ";
    }
    std::cout << "\n";

    std::vector<float> y_true = {1.0f, 0.0f};
    layer.train(input, output, y_true, 0.1f);
}
