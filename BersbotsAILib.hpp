#pragma once

#include <vector>
#include <random>
#include <cmath>

class Neuron {
public:
    Neuron(int inputSize);
    void setUseSigmoid(bool flag);
    bool getUseSigmoid() const;
    float forward(const std::vector<float>& input);
    float AIloss(float y_pred, float y_true);
    void train(const std::vector<float>& input, float y_pred, float y_true, float learning_rate);

private:
    static float sigmoidDerivative(float sigmoid_x) {
        return sigmoid_x * (1.0f - sigmoid_x);
    }
    static float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    std::vector<float> weights;
    float bias = 0;
    bool useSigmoid = 0;
};

Neuron::Neuron(int inputSize){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < inputSize; ++i) {
        weights.push_back(dist(gen));
    }
    bias = dist(gen);
}

float Neuron::forward(const std::vector<float>& input) {
    if (input.size() != weights.size()) {
        throw std::invalid_argument("Input size does not match weights size");
    }
    float sum = bias;
    for (size_t i = 0; i < weights.size(); ++i) {
        sum += weights[i] * input[i];
    }
    if (useSigmoid) {
        return sigmoid(sum);
    } else {
        return sum;
    }
}

float Neuron::AIloss(float y_pred, float y_true) {
    float diff = y_pred - y_true;
    return 0.5f * diff * diff;
}

void Neuron::setUseSigmoid(bool flag) {useSigmoid = flag;}

bool Neuron::getUseSigmoid() const {return useSigmoid;}

void Neuron::train(const std::vector<float>& input, float y_pred, float y_true, float learning_rate) {
    if (input.size() != weights.size()) {
        throw std::invalid_argument("Input size does not match weights size");
    }
    float delta;

    if (useSigmoid){
        delta = (y_pred - y_true) * sigmoidDerivative(y_pred);
    }else{
        delta = y_pred - y_true;
    }
    
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] -= learning_rate * delta * input[i];
    }

    bias -= learning_rate * delta;
}
