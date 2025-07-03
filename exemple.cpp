#include "BersbotsAILib.hpp"
#include <iostream>

int main() {
    NeuralNetwork net({2, 3, 1});

    std::vector<std::pair<std::vector<float>, std::vector<float>>> dataset = {
        { {0, 0}, {0} },
        { {0, 1}, {1} },
        { {1, 0}, {1} },
        { {1, 1}, {0} }
    };

    net.trainDataset(dataset, 10000, 0, 0.1f);

    std::cout << "Testing after training:\n";
    for (const auto& [input, target] : dataset) {
        auto output = net.predict(input);
        std::cout << input[0] << " XOR " << input[1]
                  << " = " << output[0] << " (expected " << target[0] << ")\n";
    }

    net.resetTrain();
    
    std::cout << "\n\n";

    net.trainDataset(dataset, 10, 0, 0.1f);

    std::cout << "Testing after training:\n";
    for (const auto& [input, target] : dataset) {
        auto output = net.predict(input);
        std::cout << input[0] << " XOR " << input[1]
                  << " = " << output[0] << " (expected " << target[0] << ")\n";
    }

    return 0;
}
