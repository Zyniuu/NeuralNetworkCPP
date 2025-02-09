/**
 * C++ neural network library
 *
 * DenseLayer.cpp
 */

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
        Matrix output = input * m_weights + m_biases;

        // Apply the activation function if it exists
        if (m_activation)
            output = m_activation->forward(output);

        return output;
    }

    Matrix DenseLayer::backward(const Matrix &gradient, Optimizer &optimizer)
    {
        // Compute the gradient with respect to the activation function
        Matrix gradInput = gradient;
        if (m_activation)
            gradInput = m_activation->backward(gradInput);

        // Compute the gradient with respect to the weights
        Matrix gradWeights = m_input.transpose() * gradInput;

        // Update weights and biases using the optimizer
        optimizer.update(m_weights, m_biases, gradWeights, gradInput);

        // Compute the gradient with respect to the input
        return gradInput * m_weights.transpose();
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
            init = std::make_unique<HeNormal>(inputSize, outputSize);
            break;

        case HE_UNIFORM:
            init = std::make_unique<HeUniform>(inputSize, outputSize);
            break;

        case XAVIER_NORMAL:
            init = std::make_unique<XavierNormal>(inputSize, outputSize);
            break;

        case XAVIER_UNIFORM:
            init = std::make_unique<XavierUniform>(inputSize, outputSize);
            break;
        }

        // Initialize weights using the initializer and biases to zero
        m_weights = Matrix(inputSize, outputSize, [&init]() { return init->getRandomNum(); });
        m_biases = Matrix(1, outputSize, 0.0);
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