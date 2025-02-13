/**
 * C++ neural network library
 *
 * BatchNormalization.cpp
 */

#include "BatchNormalization.hpp"
#include <cmath>

namespace nn
{
    BatchNormalization::BatchNormalization(const int numFeatures, const double momentum, const double epsilon)
        : m_momentum(momentum), m_epsilon(epsilon), m_isTraining(true), m_runningMean(0.0), m_runningVar(1.0)
    {
        m_gamma = Matrix(numFeatures, 1, 1.0);
        m_beta = Matrix(numFeatures, 1, 0.0);
        resetGradients();
    }

    BatchNormalization::BatchNormalization(std::ifstream &file)
        : m_isTraining(true)
    {
        // Check if the file is open and readable
        if (!file.is_open())
            throw std::runtime_error("File is not open for reading");

        // Read running mean and running variance
        file.read(reinterpret_cast<char *>(&m_runningMean), sizeof(m_runningMean));
        file.read(reinterpret_cast<char *>(&m_runningVar), sizeof(m_runningVar));

        // Read gamma and beta
        m_gamma = Matrix(file);
        m_beta = Matrix(file);

        // Check if reading was successful
        if (!file.good())
            throw std::runtime_error("Failed to read layer from the file.");
    }

    Matrix BatchNormalization::forward(const Matrix &input)
    {
        // Store the input
        m_input = input;

        if (m_isTraining)
        {
            // Training mode: use batch statistics
            // Calculate mean and standard deviation
            m_mean = input.sum() / input.getRows();
            m_stddev = input.map([this](double x) { return std::pow(x - m_mean, 2); }).sum() / input.getRows();

            // Update running mean and variance
            m_runningMean = m_momentum * m_runningMean + (1.0 - m_momentum) * m_mean;
            m_runningVar = m_momentum * m_runningVar + (1.0 - m_momentum) * m_stddev;

            // Normalize the input
            m_normalized = (input - m_mean) / std::sqrt(m_stddev + m_epsilon);
        }
        else
        {
            // Inference mode: use running statistics
            m_normalized = (input - m_runningMean) / std::sqrt(m_runningVar + m_epsilon);
        }

        // Scale and shift
        return m_gamma.cwiseProduct(m_normalized) + m_beta;
    }

    Matrix BatchNormalization::backward(const Matrix &gradient)
    {
        // Compute gradients for gamma and beta
        m_gradGamma += gradient.cwiseProduct(m_normalized);
        m_gradBeta += gradient;

        int m = m_input.getRows();
        double t = 1.0 / std::sqrt(m_stddev + m_epsilon); // 1 / sigma
        Matrix diff = m_input - m_mean;                   // (x_i - mu)
        Matrix gradDiff = gradient.cwiseProduct(diff);    // (dL/dy_i) * (x_i - mu)

        // Sum of gradients and gradient differences
        double sumGrad = gradient.sum();     // sum(dL/dy_j)
        double sumGradDiff = gradDiff.sum(); // sum((dL/dy_j) * (x_j - mu))

        // Compute input gradient
        Matrix gradInput = (m_gamma * t / m).cwiseProduct(m * gradient - sumGrad - (t * t) * diff * sumGradDiff);

        return gradInput;
    }

    void BatchNormalization::resetGradients()
    {
        m_gradGamma = Matrix(m_gamma.getRows(), m_gamma.getCols(), 0.0);
        m_gradBeta = Matrix(m_beta.getRows(), m_beta.getCols(), 0.0);
    }

    void BatchNormalization::applyGradient(Optimizer &optimizer, const int batchSize)
    {
        // Average gradients over the batch
        m_gradGamma /= batchSize;
        m_gradBeta /= batchSize;

        // Update gamma and beta
        optimizer.update(m_gamma, m_beta, m_gradGamma, m_gradBeta);
    }

    void BatchNormalization::save(std::ofstream &file) const
    {
        // Check if the file is open and writable
        if (!file.is_open())
            throw std::runtime_error("File is not open for writing.");

        // Save running mean and running variance
        file.write(reinterpret_cast<const char *>(&m_runningMean), sizeof(m_runningMean));
        file.write(reinterpret_cast<const char *>(&m_runningVar), sizeof(m_runningVar));

        // Save gamma and beta
        m_gamma.save(file);
        m_beta.save(file);

        // Check if writing was successful
        if (!file.good())
            throw std::runtime_error("Failed to write layer data to the file.");
    }
}