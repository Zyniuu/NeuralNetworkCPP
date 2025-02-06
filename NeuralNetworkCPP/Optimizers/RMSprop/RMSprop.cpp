/**
 * C++ neural network library
 *
 * RMSprop.cpp
 */

#include "RMSprop.hpp"
#include <cmath>

namespace nn
{
    RMSprop::RMSprop(double learningRate, double gamma, double epsilon)
        : Optimizer(learningRate), m_gamma(gamma), m_epsilon(epsilon) {}
    
    void RMSprop::update(Matrix &weights, Matrix &biases, const Matrix &gradWeights, const Matrix &gradBiases)
    {
        // Initialize moving averages if they don't exist
        if (m_vWeights.find(&weights) == m_vWeights.end())
            m_vWeights[&weights] = Matrix(weights.getRows(), weights.getCols(), 0.0);
        if (m_vBiases.find(&biases) == m_vBiases.end())
            m_vBiases[&biases] = Matrix(biases.getRows(), biases.getCols(), 0.0);

        // Update moving averages: v_t = gamma * v_{t-1} + (1 - gamma) * grad^2
        m_vWeights[&weights] = m_gamma * m_vWeights[&weights] + (1 - m_gamma) * gradWeights.cwiseProduct(gradWeights);
        m_vBiases[&biases] = m_gamma * m_vBiases[&biases] + (1 - m_gamma) * gradBiases.cwiseProduct(gradBiases);

        // Update weights and biases: w -= learing_rate * grad / (sqrt(v_t) + epsilon)
        weights -= m_learningRate * gradWeights / (m_vWeights[&weights].map([](double x) { return std::sqrt(x); }) + m_epsilon);
        biases -= m_learningRate * gradBiases / (m_vBiases[&biases].map([](double x) { return std::sqrt(x); }) + m_epsilon);
    }
}