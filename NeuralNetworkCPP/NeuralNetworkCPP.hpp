/**
 * C++ neural network library
 *
 * NeuralNetworkCPP.hpp
 */

#ifndef NEURALNETWORKCPP_HPP
#define NEURALNETWORKCPP_HPP

#include "Layers/Layers.hpp"
#include "Optimizers/Optimizers.hpp"
#include "Losses/Losses.hpp"
#include <thread>

namespace nn
{
    /**
     * @class NeuralNetworkCPP
     * @brief A neural network model.
     *
     * It supports adding layers, compiling, training, and saving/loading.
     */
    class NeuralNetworkCPP
    {
    private:
        std::vector<std::unique_ptr<Layer>> m_layers; ///< Vector of layers in the network.
        std::unique_ptr<Optimizer> m_optimizer;       ///< Optimizer for training.
        std::unique_ptr<Loss> m_loss;                 ///< Loss function for training.

    public:
        /**
         * @brief Default constructor.
         *
         * @param numThreads Number of threads in the thread pool.
         */
        NeuralNetworkCPP(int numThreads = std::thread::hardware_concurrency());

        /**
         * @brief Constructs a model from a saved file.
         *
         * @param filename Path to the file containing the saved model.
         * @param numThreads Number of threads in the thread pool.
         * @throws std::runtime_error If the file cannot be opened or is invalid.
         */
        NeuralNetworkCPP(const std::string &filename, int numThreads = std::thread::hardware_concurrency());

        /**
         * @brief Adds a layer to the network.
         *
         * @param layer The layer to add.
         */
        void addLayer(std::unique_ptr<Layer> layer);

        /**
         * @brief Compiles the model with the specified optimizer and loss function.
         *
         * @param optimizer The optimizer to use for training.
         * @param lossFunc The loss function to use for training.
         */
        void compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<Loss> lossFunc);

        /**
         * @brief Trains the model on the provided data.
         *
         * @param xTrain Training data (vector of input vectors).
         * @param yTrain Training labels (vector of output vectors).
         * @param epochs Number of training epochs.
         * @param batchSize Size of each training batch.
         * @param validationSplit Fraction of the data to use for validation.
         */
        void train(
            const std::vector<std::vector<double>> &xTrain,
            const std::vector<std::vector<double>> &yTrain,
            int epochs,
            int batchSize,
            double validationSplit
        );

        /**
         * @brief Saves the model to a file.
         *
         * @param filename Path to the file where the model will be saved.
         * @throws std::runtime_error If the file cannot be opened or writing fails.
         */
        void save(const std::string &filename) const;

    private:
        /**
         * @brief Performs forward propagation through the network.
         *
         * @param input The input matrix.
         * @return The output matrix.
         */
        Matrix forward(const Matrix &input);

        /**
         * @brief Performs backward propagation through the network.
         *
         * @param gradient The gradient of the loss with respect to the output.
         */
        void backward(const Matrix &gradient);
    };
}

#endif