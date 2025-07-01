#pragma once

#include <vector>
#include <random>
#include <cmath>

#include <stdexcept>

class AIbot {
public:
    AIbot(int inputSize, int outputSize, bool useSigmoid = true)
        : inputSize(inputSize), outputSize(outputSize), useSigmoid(useSigmoid) 
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        weights.resize(outputSize, std::vector<float>(inputSize));
        biases.resize(outputSize);

        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] = dist(gen);
            }
            biases[i] = dist(gen);
        }
    }

    std::vector<float> forward(const std::vector<float>& input) const {
        if (input.size() != inputSize) {
            throw std::invalid_argument("Input size does not match");
        }

        std::vector<float> output(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            float sum = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                sum += weights[i][j] * input[j];
            }
            output[i] = useSigmoid ? sigmoid(sum) : sum;
        }
        return output;
    }

    void train(const std::vector<float>& input, const std::vector<float>& y_pred, const std::vector<float>& y_true, float learning_rate) {
        if (y_pred.size() != outputSize || y_true.size() != outputSize) {
            throw std::invalid_argument("Output size mismatch");
        }

        for (int i = 0; i < outputSize; ++i) {
            float error = y_pred[i] - y_true[i];
            float delta = useSigmoid ? error * sigmoidDerivative(y_pred[i]) : error;

            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] -= learning_rate * delta * input[j];
            }
            biases[i] -= learning_rate * delta;
        }
    }

    float AIloss(const std::vector<float>& y_pred, const std::vector<float>& y_true) const {
        if (y_pred.size() != y_true.size()) {
            throw std::invalid_argument("Loss: output size mismatch");
        }
        float sum = 0;
        for (size_t i = 0; i < y_pred.size(); ++i) {
            float diff = y_pred[i] - y_true[i];
            sum += 0.5f * diff * diff;
        }
        return sum;
    }

    void setUseSigmoid(bool flag) {
        useSigmoid = flag;
    }

private:
    int inputSize;
    int outputSize;
    bool useSigmoid;
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;

    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    static float sigmoidDerivative(float sigmoid_x) {
        return sigmoid_x * (1.0f - sigmoid_x);
    }
};
