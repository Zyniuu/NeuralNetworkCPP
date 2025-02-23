/**
 * C++ neural network library
 *
 * ModelTrainer.hpp
 */

#include "../ModelEvaluator/ModelEvaluator.hpp"
#include "../../Losses/Losses.hpp"
#include "../../Optimizers/Optimizers.hpp"

namespace nn
{
    /**
     * @class ModelTrainer
     * @brief Handles the training of a neural network model.
     *
     * This class extends ModelEvaluator and provides functionality for compiling and training
     * the model using an optimizer, loss function, and metrics.
     */
    class ModelTrainer : public ModelEvaluator
    {
    private:
        std::unique_ptr<Optimizer> m_optimizer; ///< Optimizer for training.
        std::unique_ptr<Loss> m_loss;           ///< Loss function for training.
        std::unique_ptr<Logger> m_logger;       ///< Logger for logging training progress.
        std::vector<e_metric> m_metrics;        ///< Metrics to compute

    public:
        /**
         * @brief Performs backward propagation through the network.
         *
         * @param gradient The gradient of the loss with respect to the output.
         */
        void backward(const Matrix &gradient);

        /**
         * @brief Compiles the model with the specified optimizer and loss function.
         *
         * @param optimizer The optimizer to use for training.
         * @param lossFunc The loss function to use for training.
         * @param metrics The vector of metrics to display (default: {}).
         */
        void compile(
            std::unique_ptr<Optimizer> optimizer,
            std::unique_ptr<Loss> lossFunc,
            const std::vector<e_metric> &metrics = {}
        );

        /**
         * @brief Trains the model on the provided data.
         *
         * @param xTrain Training data (vector of input vectors).
         * @param yTrain Training labels (vector of output vectors).
         * @param epochs Number of training epochs.
         * @param batchSize Size of each training batch (default: 1).
         * @param validationSplit Fraction of the data to use for validation (default: 0.0).
         * @param patience Number of epochs to wait for improvement (default: 10).
         * @param minDelta Minimum improvement to reset patience (default: 0.0001).
         * @param verbose If true, logs will be displayed (default: true).
         * @return True if the training has been completed, false if stopped early
         */
        bool train(
            const std::vector<std::vector<double>> &xTrain,
            const std::vector<std::vector<double>> &yTrain,
            const int epochs,
            const int batchSize = 1,
            const double validationSplit = 0.0,
            const int patience = 10,
            const double minDelta = 0.0001,
            const bool verbose = true
        );

    private:
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