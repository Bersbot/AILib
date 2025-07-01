#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <stdexcept>

class Layer {
public:
    Layer(int inputSize, int outputSize) 
        : weights(outputSize, std::vector<float>(inputSize)),
          biases(outputSize)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] = dist(gen);
            }
            biases[i] = dist(gen);
        }
    }

    // Прямой проход слоя (активация сигмоида)
    std::vector<float> forward(const std::vector<float>& input) {
        if (input.size() != weights[0].size()) {
            throw std::invalid_argument("Input size does not match layer input size");
        }

        std::vector<float> output(weights.size(), 0.0f);

        for (size_t i = 0; i < weights.size(); ++i) {
            float sum = biases[i];
            for (size_t j = 0; j < weights[i].size(); ++j) {
                sum += weights[i][j] * input[j];
            }
            output[i] = sigmoid(sum);
        }

        return output;
    }

private:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;

    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
};

class NeuralNetwork {
public:
    // Конструктор принимает вектор с размерами слоев (входной, скрытые..., выходной)
    NeuralNetwork(const std::vector<int>& layersSizes) {
        if (layersSizes.size() < 2) {
            throw std::invalid_argument("Neural network must have at least 2 layers (input and output)");
        }
        for (size_t i = 1; i < layersSizes.size(); ++i) {
            layers.emplace_back(layersSizes[i - 1], layersSizes[i]);
        }
    }

    // Прямой проход по всем слоям
    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> out = input;
        for (auto& layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }

private:
    std::vector<Layer> layers;
};
