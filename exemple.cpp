#include <iostream>
#include "BersbotsAILib.hpp"

int main() {
    Neuron n(2);  // нейрон с 2 входами
    float learning_rate = 0.1f;

    // Обучающие данные для OR
    std::vector<std::pair<std::vector<float>, float>> dataset = {
        {{0, 0}, 0},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, 1}
    };

    // Цикл обучения
    for (int epoch = 0; epoch < 1000; ++epoch) {
        float total_loss = 0.0f;
        for (auto& sample : dataset) {
            auto input = sample.first;
            float target = sample.second;

            float output = n.forward(input);
            float loss = n.AIloss(output, target);
            total_loss += loss;

            n.train(input, output, target, learning_rate);
        }

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss = " << total_loss << std::endl;
        }
    }

    // Проверим результат
    std::cout << "Test after training:\n";
    for (auto& sample : dataset) {
        float out = n.forward(sample.first);
        std::cout << sample.first[0] << " OR " << sample.first[1] << " = " << out << std::endl;
    }

    return 0;
}
