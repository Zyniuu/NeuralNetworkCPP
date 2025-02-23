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
#include "ModelParts/ModelTrainer/ModelTrainer.hpp"
#include <thread>

namespace nn
{
    /**
     * @class NeuralNetworkCPP
     * @brief A neural network model that supports adding layers, compiling, training, and saving/loading.
     *
     * This class provides a high-level interface for building and training neural networks.
     */
    class NeuralNetworkCPP : public ModelTrainer
    {
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
         * @brief Saves the model to a file.
         *
         * @param filename Path to the file where the model will be saved.
         * @throws std::runtime_error If the file cannot be opened or writing fails.
         */
        void save(const std::string &filename) const;
    };
}

#endif
