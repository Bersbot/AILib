#pragma once

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

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

    void initialize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (auto& neuron_weights : weights) {
            for (float& w : neuron_weights) {
                w = dist(gen);
            }
        }

        for (float& b : biases) {
            b = dist(gen);
        }
    }


    const std::vector<float>& getOutput() const {
        return lastOutput;
    }

    const std::vector<std::vector<float>>& getWeights() const {
        return weights;
    }

    const std::vector<float>& getBiases() const {
        return biases;
    }

    std::vector<std::vector<float>>& getWeights() {
        return weights;
    }

    std::vector<float>& getBiases() {
        return biases;
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

    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> out = input;
        for (auto& layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }

    void train(const std::vector<float>& input, const std::vector<float>& target, float learningRate) {
        if (target.size() != layers.back().getOutput().size()) {
            throw std::invalid_argument("Target size must match output layer size");
        }

        std::vector<std::vector<float>> activations;
        activations.push_back(input);
        std::vector<float> out = input;
        for (auto& layer : layers) {
            out = layer.forward(out);
            activations.push_back(out);
        }

        std::vector<float> delta(out.size());
        for (size_t i = 0; i < out.size(); ++i) {
            delta[i] = out[i] - target[i];
        }

        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            delta = layers[i].backward(delta, activations[i], learningRate);
        }
    }  
    static float mse(const std::vector<float>& output, const std::vector<float>& target) {
        float sum = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            float diff = output[i] - target[i];
            sum += diff * diff;
        }
        return sum / output.size();
    }

    void trainDataset(const std::vector<std::pair<std::vector<float>, std::vector<float>>>& dataset, int epochs, bool showProgres, float learningRate) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float totalLoss = 0.0f;
            for (const auto& [input, target] : dataset) {
                auto output = forward(input);
                if (target.size() != output.size()) {
                    std::cerr << "Error: target size (" << target.size()
                              << ") != output size (" << output.size() << ")\n";
                    throw std::invalid_argument("Target size does not match output layer size");
                }
                train(input, target, learningRate);
                totalLoss += mse(output, target);
            }
            if (showProgres){std::cout << "Epoch " << epoch << ", loss = " << totalLoss / dataset.size() << std::endl;}
        }
    }


    std::vector<float> predict(const std::vector<float>& input) {
        return forward(input);
    }

    std::vector<float> softmax(const std::vector<float>& x) {
        std::vector<float> result(x.size());
        float maxVal = *std::max_element(x.begin(), x.end());
        float sum = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - maxVal);
            sum += result[i];
        }
        for (float& v : result) v /= sum;
        return result;
    }

     void saveWeights(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file) {
            throw std::runtime_error("Cannot open file for saving weights");
        }

        file << layers.size() << "\n";
        for (const auto& layer : layers) {
            file << layer.getWeights().size() << " " << layer.getWeights()[0].size() << "\n";
            for (const auto& neuron_weights : layer.getWeights()) {
                for (float w : neuron_weights) {
                    file << w << " ";
                }
                file << "\n";
            }
            for (float b : layer.getBiases()) {
                file << b << " ";
            }
            file << "\n";
        }
    }

    // Загружаем веса и смещения из файла
    void loadWeights(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Cannot open file for loading weights");
        }

        size_t numLayers = 0;
        file >> numLayers;
        if (numLayers != layers.size()) {
            throw std::runtime_error("Mismatch in number of layers when loading weights");
        }

        for (auto& layer : layers) {
            size_t outputSize = 0, inputSize = 0;
            file >> outputSize >> inputSize;
            if (outputSize != layer.getWeights().size() || inputSize != layer.getWeights()[0].size()) {
                throw std::runtime_error("Mismatch in layer sizes when loading weights");
            }

            for (auto& neuron_weights : layer.getWeights()) {
                for (float& w : neuron_weights) {
                    file >> w;
                }
            }
            for (float& b : layer.getBiases()) {
                file >> b;
            }
        }
    }

    void resetTrain() {
        for (auto& layer : layers) {
            layer.initialize();
        }
    }



private:
    std::vector<Layer> layers;
};
