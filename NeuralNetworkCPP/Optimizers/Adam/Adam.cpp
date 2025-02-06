/**
 * C++ neural network library
 *
 * Adam.cpp
 */

#include "Adam.hpp"
#include <cmath>

namespace nn
{
    Adam::Adam(double learningRate, double beta1, double beta2, double epsilon)
        : Optimizer(learningRate), m_beta1(beta1), m_beta2(beta2), m_epsilon(epsilon), m_t(0) {}

    void Adam::update(Matrix &weights, Matrix &biases, const Matrix &gradWeights, const Matrix &gradBiases)
    {
        // Initialize moment estimates if they don't exist
        if (m_m.find(&weights) == m_m.end())
        {
            m_m[&weights] = Matrix(weights.getRows(), weights.getCols(), 0.0);
            m_v[&weights] = Matrix(weights.getRows(), weights.getCols(), 0.0);
        }
        if (m_m.find(&biases) == m_m.end())
        {
            m_m[&biases] = Matrix(biases.getRows(), biases.getCols(), 0.0);
            m_v[&biases] = Matrix(biases.getRows(), biases.getCols(), 0.0);
        }

        // Increase time step
        m_t++;

        // Update first moment estimates: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
        m_m[&weights] = m_beta1 * m_m[&weights] + (1.0 - m_beta1) * gradWeights;
        m_m[&biases] = m_beta1 * m_m[&biases] + (1.0 - m_beta1) * gradBiases;

        // Update second moment estimates: v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
        m_v[&weights] = m_beta2 * m_v[&weights] + (1.0 - m_beta2) * gradWeights.cwiseProduct(gradWeights);
        m_v[&biases] = m_beta2 * m_v[&biases] + (1.0 - m_beta2) * gradBiases.cwiseProduct(gradBiases);

        // Bias-corrected first moment estimates
        double mHat = 1.0 / (1.0 - std::pow(m_beta1, m_t));
        Matrix mHatWeights = m_m[&weights] * mHat;
        Matrix mHatBiases = m_m[&biases] * mHat;

        // Bias-corrected second moment estimates
        double vHat = 1.0 / (1.0 - std::pow(m_beta2, m_t));
        Matrix vHatWeights = m_v[&weights] * vHat;
        Matrix vHatBiases = m_v[&biases] * vHat;

        // Update weights and biases
        weights -= m_learningRate * mHatWeights / (vHatWeights.map([](double x) { return std::sqrt(x); }) + m_epsilon);
        biases -= m_learningRate * mHatBiases / (vHatBiases.map([](double x) { return std::sqrt(x); }) + m_epsilon);
    }
}