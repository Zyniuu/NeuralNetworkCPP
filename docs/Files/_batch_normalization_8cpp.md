# Layers/BatchNormalization/BatchNormalization.cpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |




## Source code

```cpp


#include "BatchNormalization.hpp"
#include <cmath>

namespace nn
{
    BatchNormalization::BatchNormalization(const int numFeatures, const double momentum, const double epsilon)
        : m_momentum(momentum), m_epsilon(epsilon), m_isTraining(true)
    {
        m_gamma = Matrix(numFeatures, 1, 1.0);
        m_beta = Matrix(numFeatures, 1, 0.0);
        m_runningMean = Matrix(numFeatures, 1, 0.0);
        m_runningVar = Matrix(numFeatures, 1, 0.0);
        resetGradients();
    }

    BatchNormalization::BatchNormalization(std::ifstream &file)
        : m_isTraining(true)
    {
        // Check if the file is open and readable
        if (!file.is_open())
            throw std::runtime_error("File is not open for reading");

        // Read momentum
        file.read(reinterpret_cast<char *>(&m_momentum), sizeof(m_momentum));

        // Read epsilon
        file.read(reinterpret_cast<char *>(&m_epsilon), sizeof(m_epsilon));

        // Read running mean and running variance
        m_runningMean = Matrix(file);
        m_runningVar = Matrix(file);

        // Read gamma and beta
        m_gamma = Matrix(file);
        m_beta = Matrix(file);

        // Check if reading was successful
        if (!file.good())
            throw std::runtime_error("Failed to read layer from the file.");

        resetGradients();
    }

    Matrix BatchNormalization::forward(const Matrix &input)
    {
        // Store the input
        m_input = input;

        if (m_isTraining)
        {
            // Training mode: use batch statistics
            // Calculate mean and standard deviation
            m_mean = m_input.rowWise().sum() / input.getCols();
            Matrix diff = m_input.colWise() - m_mean;
            m_stddev = diff.cwiseProduct(diff).rowWise().sum() / m_input.getCols();

            // Update running mean and variance
            m_runningMean = m_momentum * m_runningMean + (1.0 - m_momentum) * m_mean;
            m_runningVar = m_momentum * m_runningVar + (1.0 - m_momentum) * m_stddev;

            // Normalize the input
            m_normalized = diff.colWise() / (m_stddev + m_epsilon).map([](double x) { return std::sqrt(x); });
        }
        else
        {
            // Inference mode: use running statistics
            Matrix diff = m_input.colWise() - m_runningMean;
            m_normalized = diff.colWise() / (m_runningVar + m_epsilon).map([](double x) { return std::sqrt(x); });
        }

        // Scale and shift
        return (m_normalized.colWise() * m_gamma).colWise() + m_beta;
    }

    Matrix BatchNormalization::backward(const Matrix &gradient)
    {
        int m = m_input.getCols();
        Matrix t = (m_stddev + m_epsilon).map([](double x) { return 1.0 / std::sqrt(x); }); // 1 / sigma
        Matrix diff = m_input.colWise() - m_mean;                                           // (x_i - mu)
        Matrix gradDiff = diff.cwiseProduct(gradient);                                      // (dL/dy_i) * (x_i - mu)

        // Sum of gradients
        Matrix sumGrad = gradient.rowWise().sum(); // sum(dL/dy_j)

        // Compute input gradient
        Matrix gradInput = (((m * gradient).colWise() - sumGrad) - ((diff.colWise() * t.cwiseProduct(t)).colWise() * gradDiff.rowWise().sum())).colWise() * (m_gamma.cwiseProduct(t) / m);

        // Compute gradients for gamma and beta
        m_gradGamma += gradient.cwiseProduct(m_normalized).rowWise().sum();
        m_gradBeta += sumGrad;

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

        // Save momentum
        file.write(reinterpret_cast<const char *>(&m_momentum), sizeof(m_momentum));

        // Save epsilon
        file.write(reinterpret_cast<const char *>(&m_epsilon), sizeof(m_epsilon));

        // Save running mean and running variance
        m_runningMean.save(file);
        m_runningVar.save(file);

        // Save gamma and beta
        m_gamma.save(file);
        m_beta.save(file);

        // Check if writing was successful
        if (!file.good())
            throw std::runtime_error("Failed to write layer data to the file.");
    }
}
```
