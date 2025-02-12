/**
 * C++ neural network library
 *
 * BatchNormalization.cpp
 */

#include "BatchNormalization.hpp"
#include <cmath>

namespace nn
{
    BatchNormalization::BatchNormalization(const int numFeatures, const double epsilon, const double momentum)
        : m_epsilon(epsilon), m_momentum(momentum), m_isTraining(true), m_runningMean(0.0), m_runningVar(1.0)
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
        if (m_isTraining)
        {
            // Training mode: use batch statistics
            // Calculate mean and standard deviation
            double mean = input.sum() / input.getRows();
            double stddev = input.map([mean](double x) { return std::pow(x - mean, 2); }).sum() / input.getRows();

            // Update running mean and variance
            m_runningMean = m_momentum * m_runningMean + (1.0 - m_momentum) * mean;
            m_runningVar = m_momentum * m_runningVar + (1.0 - m_momentum) * stddev;
            
            // Normalize the input
            m_normalized = (input - mean) / std::sqrt(stddev + m_epsilon);
        }
        else
        {
            // Inference mode: use running statistics
            m_normalized = (input - m_runningMean) / std::sqrt(m_runningVar + m_epsilon);
        }

        // std::cout << "Input:" << std::endl;
        // std::cout << input << std::endl;
        // std::cout << "m_gamma:" << std::endl;
        // std::cout << m_gamma << std::endl;
        // std::cout << "m_beta:" << std::endl;
        // std::cout << m_beta << std::endl;
        // std::cout << "m_normalized:" << std::endl;
        // std::cout << m_normalized << std::endl;
        // std::cout << "Scale and shift:" << std::endl;
        // std::cout << (m_gamma.cwiseProduct(m_normalized) + m_beta) << std::endl;
        // getchar();

        // Scale and shift
        return m_gamma.cwiseProduct(m_normalized) + m_beta;
    }

    Matrix BatchNormalization::backward(const Matrix &gradient)
    {
        // Compute gradients for gamma and beta
        m_gradGamma += gradient.cwiseProduct(m_normalized);
        m_gradBeta += gradient;

        // Compute input gradient
        Matrix gradNormalized = gradient.cwiseProduct(m_gamma);
        return gradNormalized / std::sqrt(m_runningVar + m_epsilon);
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