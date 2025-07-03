# BersbotsAILib

**BersbotsAILib** is a simple, self-contained (header-only) utility library in C++17+ for building and training neural networks **without external dependencies**.

---

## 🔧 Features

- Create neural networks with custom architectures: `NeuralNetwork({input, hidden..., output})`
- Training using **backpropagation**
- Predictions: `predict(...)`
- Resetting weights: `resetTrain()`
- Saving and loading weights: `saveWeights(...)` and `loadWeights(...)`
- Uses **sigmoid** activation function
- Works with custom datasets

---

## Example Usage

```cpp
#include "BersbotsAILib.hpp"
#include <iostream>

int main() {
    NeuralNetwork net({2, 3, 1}); // network: 2 inputs → 3 hidden → 1 output

    std::vector<std::pair<std::vector<float>, std::vector<float>>> dataset = {
        { {0, 0}, {0} },
        { {0, 1}, {1} },
        { {1, 0}, {1} },
        { {1, 1}, {0} }
    };

    net.trainDataset(dataset, 10000, true, 0.1f);

    for (const auto& [input, target] : dataset) {
        auto output = net.predict(input);
        std::cout << input[0] << " XOR " << input[1]
                  << " = " << output[0]
                  << " (expected " << target[0] << ")\n";
    }

    net.saveWeights("weights.txt");
    net.resetTrain();
    net.loadWeights("weights.txt");

    return 0;
}
