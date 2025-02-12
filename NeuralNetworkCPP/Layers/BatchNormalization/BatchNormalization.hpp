/**
 * C++ neural network library
 *
 * BatchNormalization.hpp
 */

#ifndef BATCHNORMALIZATION_HPP
#define BATCHNORMALIZATION_HPP

#include "../Common/Layer.hpp"

namespace nn
{
    /**
     * @class BatchNormalization
     * @brief Implements Batch Normalization layer.
     *
     * Batch Normalization is a technique used to normalize the inputs
     * of each layer to improve the training speed and stability of neural networks.
     */
    class BatchNormalization : public Layer
    {
    private:
        Matrix m_gamma;       ///< Scale parameter (learnable).
        Matrix m_beta;        ///< Shift parameter (learnable).
        Matrix m_normalized;  ///< Normalized input (stored for backward pass).
        Matrix m_gradGamma;   ///< Accumulated gradient for gamma.
        Matrix m_gradBeta;    ///< Accumulated gradient for beta.
        double m_runningMean; ///< Running mean (used during inference).
        double m_runningVar;  ///< Running variance (used during inference).
        double m_epsilon;     ///< Small constant for numerical stability.
        double m_momentum;    ///< Momentum for updating running mean and variance.
        bool m_isTraining;    ///< Flag to indicate whether the layer is in training mode.

    public:
        /**
         * @brief Constructs a BatchNormalization layer.
         *
         * @param numFeatures Number of features (input dimensions).
         * @param epsilon Small constant for numerical stability.
         * @param momentum Momentum for updating running mean and variance.
         */
        BatchNormalization(const int numFeatures, const double epsilon = 1e-15, const double momentum = 0.9);

        /**
         * @brief Constructs a BatchNormalization layer from a file.
         *
         * @param file Input file stream (must be opened in binary mode).
         * @throws std::runtime_error If the file is not open or reading fails.
         */
        BatchNormalization(std::ifstream &file);

        /**
         * @brief Performs forward propagation.
         *
         * @param input The input matrix.
         * @return The output matrix after applying BatchNormalization.
         */
        Matrix forward(const Matrix &input) override;

        /**
         * @brief Performs backward propagation.
         *
         * @param gradient The gradient of the loss with respect to the output.
         * @return The gradient of the loss with respect to the input.
         */
        Matrix backward(const Matrix &gradient) override;

        /**
         * @brief Resets the accumulated gradients of the layer.
         */
        void resetGradients() override;

        /**
         * @brief Applies accumulated gradients to the layer.
         *
         * @param optimizer The optimizer to use for gamma and beta updates.
         * @param batchSize Size of the batch from which gradients were accumulated.
         */
        void applyGradient(Optimizer &optimizer, const int batchSize) override;

        /**
         * @brief Saves the layer's state to a binary file.
         *
         * @param file Output file stream (must be opened in binary mode).
         * @throws std::runtime_error If the file is not open or reading fails.
         */
        void save(std::ofstream &file) const override;

        /**
         * @brief Returns the type of the layer.
         *
         * @return The layer type as an enum value.
         */
        e_layerType getType() const override { return BATCH_NORM; };

        /**
         * @brief Sets the layer's training mode.
         *
         * @param isTraining True for training, false for inference.
         */
        void setTrainingMode(const bool isTrainging) { m_isTraining = isTrainging; };
    };
}

#endif