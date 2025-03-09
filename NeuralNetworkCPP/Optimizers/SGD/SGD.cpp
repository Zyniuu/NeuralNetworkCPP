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
        if (m_velocities.find(&weights) == m_velocities.end())
            m_velocities[&weights] = Matrix(weights.getRows(), weights.getCols(), 0.0);
        if (m_velocities.find(&biases) == m_velocities.end())
            m_velocities[&biases] = Matrix(biases.getRows(), biases.getCols(), 0.0);

        // Update velocity for weights: v_t = momentum * v_{t-1} + learingRate * gradWeights
        m_velocities[&weights] = (m_momentum * m_velocities[&weights]) + (m_learningRate * gradWeights);

        // Update velocity for biases: v_t = momentum * v_{t-1} + learingRate * gradBiases
        m_velocities[&biases] = (m_momentum * m_velocities[&biases]) + (m_learningRate * gradBiases);

        // Update weights and biases
        weights -= m_velocities[&weights];
        biases -= m_velocities[&biases];
    }
}