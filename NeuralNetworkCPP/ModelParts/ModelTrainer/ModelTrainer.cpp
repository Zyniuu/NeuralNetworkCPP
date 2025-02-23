/**
 * C++ neural network library
 *
 * ModelTrainer.cpp
 */

#include "ModelTrainer.hpp"
#include "../../Utils/Utils.hpp"
#include <cmath>

namespace nn
{
    void ModelTrainer::backward(const Matrix &gradient)
    {
        // Propagate the gradient backward through the layers
        Matrix grad = gradient;

        for (auto it = m_layers.rbegin(); it != m_layers.rend(); it++)
            grad = (*it)->backward(grad);
    }

    void ModelTrainer::compile(
        std::unique_ptr<Optimizer> optimizer,
        std::unique_ptr<Loss> lossFunc,
        const std::vector<e_metric> &metrics
    )
    {
        // Set the optimizer, loss function, logger and metrics
        m_optimizer = std::move(optimizer);
        m_loss = std::move(lossFunc);
        m_logger = std::make_unique<Logger>();
        m_metrics = metrics;
    }

    bool ModelTrainer::train(
        const std::vector<std::vector<double>> &xTrain,
        const std::vector<std::vector<double>> &yTrain,
        const int epochs,
        const int batchSize,
        const double validationSplit,
        const int patience,
        const double minDelta,
        const bool verbose
    )
    {
        double bestLoss = std::numeric_limits<double>::max();
        int waitCounter = 0;

        // Copy the dataset
        std::vector<std::vector<double>> xTrainCopy = xTrain;
        std::vector<std::vector<double>> yTrainCopy = yTrain;

        // Shuffle the dataset
        shuffleDataset(xTrainCopy, yTrainCopy);

        // Split data into training and validation sets
        int numValidation = xTrainCopy.size() * validationSplit;
        std::vector<std::vector<double>> xValSplit = slice(xTrainCopy, 0, numValidation);
        std::vector<std::vector<double>> yValSplit = slice(yTrainCopy, 0, numValidation);
        std::vector<std::vector<double>> xTrainSplit = slice(xTrainCopy, numValidation, xTrainCopy.size());
        std::vector<std::vector<double>> yTrainSplit = slice(yTrainCopy, numValidation, yTrainCopy.size());

        // Compute total number of batches
        double totalBatches = static_cast<double>(xTrainSplit.size()) / static_cast<double>(batchSize);

        // Log training start
        if (verbose)
            m_logger->logTrainingStart();

        // Training loop
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Log epoch start
            if (verbose)
                m_logger->logEpochStart(epoch + 1, epochs);
            
            int batchIndex = 0;
            double loss = 0.0;

            // Shuffle training data before each epoch
            shuffleDataset(xTrainSplit, yTrainSplit);

            // Process batches
            for (int i = 0; i < xTrainSplit.size(); i += batchSize)
            {
                batchIndex++;

                // Log batch progress
                if (verbose)
                    m_logger->logBatch(batchIndex, std::ceil(totalBatches));

                // Make sure batch doesn't overflow
                int end = std::min(static_cast<int>(xTrainSplit.size()), i + batchSize);

                // Get the current batch
                std::vector<std::vector<double>> xBatch = slice(xTrainSplit, i, end);
                std::vector<std::vector<double>> yBatch = slice(yTrainSplit, i, end);

                // Train on the current batch
                trainOnBatch(xBatch, yBatch, loss);
            }

            // Compute average loss and other metrics
            loss /= xTrainSplit.size();
            std::vector<double> computedMetrics = evaluate(xValSplit, yValSplit, m_metrics);

            // Log epoch end
            if (verbose)
                m_logger->logEpochEnd(std::ceil(totalBatches), loss, computedMetrics, m_metrics);
            
            // Early stopping check
            if (loss < bestLoss - minDelta)
            {
                bestLoss = loss;
                waitCounter = 0;
            }
            else
            {
                waitCounter++;
                if (waitCounter >= patience)
                {
                    // Log early stop
                    if (verbose)
                        m_logger->logTrainingEnd(true);
                    return false; // Stop training
                }
            }
        }

        // Log training end
        if (verbose)
            m_logger->logTrainingEnd(false);
        
        return true;
    }

    void ModelTrainer::trainOnBatch(
        const std::vector<std::vector<double>> &xBatch,
        const std::vector<std::vector<double>> &yBatch,
        double &loss
    )
    {
        // Reset gradients for all layers
        for (auto &layer : m_layers)
            layer->resetGradients();

        // Forward and backward passes (accumulating gradients)
        // Convert the batch of inputs and targets into matrices
        Matrix inputBatch = nn::Matrix(xBatch).transpose(); // Each column is a sample
        Matrix targetBatch = nn::Matrix(yBatch).transpose(); // Each column is a target

        // Forward pass for the entire batch
        Matrix outputBatch = forward(inputBatch);

        // Compute the loss for the entire batch
        loss += m_loss->computeLoss(outputBatch, targetBatch);

        // Backward pass for the entire batch
        Matrix gradBatch = m_loss->computeGradient(outputBatch, targetBatch);
        backward(gradBatch);

        // Average gradients and update weights
        for (auto &layer : m_layers)
            layer->applyGradient(*m_optimizer, xBatch.size());
    }
}