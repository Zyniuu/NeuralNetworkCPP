/**
 * C++ neural network library
 *
 * SGD.cpp
 */

#include "SGD.hpp"

namespace nn
{
    SGD::SGD(double learningRate, double momentum)
        : Optimizer(learningRate), m_momentum(momentum) {}

    void SGD::update(Matrix &weights, Matrix &biases, const Matrix &gradWeights, const Matrix &gradBiases)
    {
        // Initialize velocity matrices if they don't exist
        if (m_velocityWeights.find(&weights) == m_velocityWeights.end())
            m_velocityWeights[&weights] = Matrix(weights.getRows(), weights.getCols(), 0.0);
        if (m_velocityBiases.find(&biases) == m_velocityBiases.end())
            m_velocityBiases[&biases] = Matrix(biases.getRows(), biases.getCols(), 0.0);

        // Update velocity for weights: v_t = momentum * v_{t-1} + learingRate * gradWeights
        m_velocityWeights[&weights] = m_momentum * m_velocityWeights[&weights] + m_learningRate * gradWeights;

        // Update velocity for biases: v_t = momentum * v_{t-1} + learingRate * gradBiases
        m_velocityBiases[&biases] = m_momentum * m_velocityBiases[&biases] + m_learningRate * gradBiases;

        // Update weights and biases
        weights -= m_velocityWeights[&weights];
        biases -= m_velocityBiases[&biases];
    }
}