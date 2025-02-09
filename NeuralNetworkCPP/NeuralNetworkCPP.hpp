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
#include "Logger/Logger.hpp"
#include <thread>

namespace nn
{
    /**
     * @class NeuralNetworkCPP
     * @brief A neural network model that supports adding layers, compiling, training, and saving/loading.
     *
     * This class provides a high-level interface for building and training neural networks.
     */
    class NeuralNetworkCPP
    {
    private:
        std::vector<std::unique_ptr<Layer>> m_layers; ///< Vector of layers in the network.
        std::unique_ptr<Optimizer> m_optimizer;       ///< Optimizer for training.
        std::unique_ptr<Loss> m_loss;                 ///< Loss function for training.
        std::unique_ptr<Logger> m_logger;             ///< Logger for logging training progress.

    public:
        /**
         * @brief Default constructor.
         *
         * @param numThreads Number of threads in the thread pool (default: hardware concurrency).
         */
        NeuralNetworkCPP(const int numThreads = std::thread::hardware_concurrency());

        /**
         * @brief Constructs a model from a saved file.
         *
         * @param filename Path to the file containing the saved model.
         * @param numThreads Number of threads in the thread pool (default: hardware concurrency).
         * @throws std::runtime_error If the file cannot be opened or is invalid.
         */
        NeuralNetworkCPP(const std::string &filename, const int numThreads = std::thread::hardware_concurrency());

        /**
         * @brief Predicts the output for a given input.
         *
         * @param input The input vector.
         * @return The predicted output vector.
         */
        std::vector<double> predict(const std::vector<double> &input);

        /**
         * @brief Evaluates the model on the provided test data.
         *
         * @param xTest Test data (vector of input vectors).
         * @param yTest Test labels (vector of output vectors).
         * @return The accuracy of the model on the test data.
         */
        double evaluate(const std::vector<std::vector<double>> &xTest, const std::vector<std::vector<double>> &yTest);

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
         * @param metrics The vector of metrics to display (default: {}).
         */
        void compile(std::unique_ptr<Optimizer> optimizer, std::unique_ptr<Loss> lossFunc, std::vector<e_metric> metrics = {});

        /**
         * @brief Trains the model on the provided data.
         *
         * @param xTrain Training data (vector of input vectors).
         * @param yTrain Training labels (vector of output vectors).
         * @param epochs Number of training epochs.
         * @param batchSize Size of each training batch (default: 1).
         * @param validationSplit Fraction of the data to use for validation (default: 0.0).
         * @param verbose If true, logs will be displayed (default: true).
         */
        void train(
            const std::vector<std::vector<double>> &xTrain,
            const std::vector<std::vector<double>> &yTrain,
            const int epochs,
            const int batchSize = 1,
            const double validationSplit = 0.0,
            const bool verbose = true
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

        /**
         * @brief Initializes a layer based on the provided layer type.
         *
         * @param layerType Enum value of the layer type.
         * @param file Input file stream (must be opened in binary mode).
         * @throws std::runtime_error If the layer type is invalid.
         */
        void initLayer(e_layerType layerType, std::ifstream &file);

        /**
         * @brief Trains the model on a single batch.
         *
         * @param xBatch Batch of input data.
         * @param yBatch Batch of target data.
         * @param loss Accumulated loss for the batch.
         */
        void trainOnBatch(
            const std::vector<std::vector<double>> &xBatch, 
            const std::vector<std::vector<double>> &yBatch, 
            double &loss
        );
    };
}

#endif
