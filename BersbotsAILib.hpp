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

    std::vector<float> forward(const std::vector<float>& input) {
        if (input.size() != weights[0].size()) {
            throw std::invalid_argument("Input size does not match layer input size");
        }

        lastInput = input;
        lastOutput.resize(weights.size());

        for (size_t i = 0; i < weights.size(); ++i) {
            float sum = biases[i];
            for (size_t j = 0; j < weights[i].size(); ++j) {
                sum += weights[i][j] * input[j];
            }
            lastOutput[i] = sigmoid(sum);
        }

        return lastOutput;
    }

    std::vector<float> backward(const std::vector<float>& delta, const std::vector<float>& prevActivation, float learningRate) {
        std::vector<float> deltaPrev(weights[0].size(), 0.0f);

        for (size_t i = 0; i < weights.size(); ++i) {
            float sigmoid_derivative = lastOutput[i] * (1 - lastOutput[i]);
            float delta_val = delta[i] * sigmoid_derivative;

            for (size_t j = 0; j < weights[i].size(); ++j) {
                deltaPrev[j] += weights[i][j] * delta_val;
                weights[i][j] -= learningRate * delta_val * prevActivation[j];
            }

            biases[i] -= learningRate * delta_val;
        }
        return deltaPrev;
    }

    const std::vector<float>& getOutput() const {
        return lastOutput;
    }

private:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;

    std::vector<float> lastInput;
    std::vector<float> lastOutput;

    static inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
};


class NeuralNetwork {
public:
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
        for (const auto& layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }

    // Заглушка для обучения — пока без реализации
    void train(const std::vector<float>& input, const std::vector<float>& target, float learningRate) {
        if (target.size() != layers.back().getOutput().size()) {
            throw std::invalid_argument("Target size must match output layer size");
        }

        // Прямой проход
        std::vector<std::vector<float>> activations; // активации всех слоев (включая входной)
        activations.push_back(input);
        std::vector<float> out = input;
        for (auto& layer : layers) {
            out = layer.forward(out);
            activations.push_back(out);
        }

        // Вычисляем ошибку выходного слоя
        std::vector<float> delta(out.size());
        for (size_t i = 0; i < out.size(); ++i) {
            delta[i] = out[i] - target[i]; // градиент ошибки по выходу (MSE)
        }

        // Обратное распространение ошибки
        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            delta = layers[i].backward(delta, activations[i], learningRate);
        }
    }

private:
    std::vector<Layer> layers;
};
