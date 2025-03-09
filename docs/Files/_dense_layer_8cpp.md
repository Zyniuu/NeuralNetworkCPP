# Layers/DenseLayer/DenseLayer.cpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |




## Source code

```cpp


#include "DenseLayer.hpp"
#include "../../Initializers/Initializers.hpp"
#include "../../Activations/Activations.hpp"

namespace nn
{
    DenseLayer::DenseLayer(const int inputSize, const int outputSize, e_initializer initializerID, e_activation activationID)
    {
        // Validate input and output sizes
        if (inputSize <= 0 || outputSize <= 0)
            throw std::invalid_argument("Input and output size of the layer must be greater than zero.");

        // Initialize weights and biases
        initWeights(inputSize, outputSize, initializerID);

        // Initialize activation function
        initActivationFunction(activationID);
    }

    DenseLayer::DenseLayer(std::ifstream &file)
    {
        // Check if the file is open and readable
        if (!file.is_open())
            throw std::runtime_error("File is not open for reading");

        // Read the activation function ID
        file.read(reinterpret_cast<char *>(&m_activationID), sizeof(m_activationID));

        // Initialize the activation function based on the ID
        initActivationFunction(m_activationID);

        // Read weights and biases from the file
        m_weights = Matrix(file);
        m_biases = Matrix(file);

        // Check if reading was successful
        if (!file.good())
            throw std::runtime_error("Failed to read layer from the file.");
    }

    Matrix DenseLayer::forward(const Matrix &input)
    {
        // Store the input for use in the backward pass
        m_input = input;

        // Compute the linear transformation: output = input * weights + biases
        m_output = (m_weights * m_input).colWise() + m_biases;

        // Apply the activation function if it exists
        Matrix output = m_output;
        if (m_activation)
            output = m_activation->forward(output);

        return output;
    }

    Matrix DenseLayer::backward(const Matrix &gradient)
    {
        // Compute the gradient with respect to the output
        Matrix gradOutput = m_activation ? m_activation->backward(m_output) : Matrix(m_output.getRows(), m_output.getCols(), 1.0);
        gradOutput = gradient.cwiseProduct(gradOutput);

        // Accumulate gradients
        m_gradWeights += gradOutput * m_input.transpose();
        m_gradBiases += gradOutput.rowWise().sum();

        // Compute the gradient with respect to the input
        return m_weights.transpose() * gradOutput;
    }

    void DenseLayer::resetGradients()
    {
        m_gradWeights = Matrix(m_weights.getRows(), m_weights.getCols(), 0.0);
        m_gradBiases = Matrix(m_biases.getRows(), m_biases.getCols(), 0.0);
    }

    void DenseLayer::applyGradient(Optimizer &optimizer, const int batchSize)
    {
        // Average gradients over the batch
        m_gradWeights /= batchSize;
        m_gradBiases /= batchSize;

        // Update weights and biases
        optimizer.update(m_weights, m_biases, m_gradWeights, m_gradBiases);
    }

    void DenseLayer::save(std::ofstream &file) const
    {
        // Check if the file is open and writable
        if (!file.is_open())
            throw std::runtime_error("File is not open for writing.");

        // Write the activation function ID to the file
        file.write(reinterpret_cast<const char *>(&m_activationID), sizeof(m_activationID));

        // Save weights and biases to the file
        m_weights.save(file);
        m_biases.save(file);

        // Check if writing was successful
        if (!file.good())
            throw std::runtime_error("Failed to write layer data to the file.");
    }

    void DenseLayer::initWeights(const int inputSize, const int outputSize, e_initializer initializerID)
    {
        // Create the appropriate initializer based on the provided ID
        std::unique_ptr<Initializer> init;

        switch (initializerID)
        {
        case HE_NORMAL:
            init = std::make_unique<HeNormal>(outputSize, inputSize);
            break;

        case HE_UNIFORM:
            init = std::make_unique<HeUniform>(outputSize, inputSize);
            break;

        case XAVIER_NORMAL:
            init = std::make_unique<XavierNormal>(outputSize, inputSize);
            break;

        case XAVIER_UNIFORM:
            init = std::make_unique<XavierUniform>(outputSize, inputSize);
            break;
        }

        // Initialize weights using the initializer and biases to zero
        m_weights = Matrix(outputSize, inputSize, [&init]() { return init->getRandomNum(); });
        m_biases = Matrix(outputSize, 1, 0.0);
    }

    void DenseLayer::initActivationFunction(e_activation activationID)
    {
        // Initialize the activation function based on the provided ID
        switch (activationID)
        {
        case RELU:
            m_activationID = RELU;
            m_activation = std::make_unique<ReLU>();
            break;

        case SIGMOID:
            m_activationID = SIGMOID;
            m_activation = std::make_unique<Sigmoid>();
            break;

        case SOFTMAX:
            m_activationID = SOFTMAX;
            m_activation = std::make_unique<Softmax>();
            break;

        case NONE:
            m_activationID = NONE;
            m_activation = nullptr;
            break;
        
        default:
            throw std::runtime_error("Invalid activation function ID.");
        }
    }
}
```
