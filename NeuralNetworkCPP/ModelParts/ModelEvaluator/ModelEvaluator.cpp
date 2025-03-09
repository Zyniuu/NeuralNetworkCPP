/**
 * C++ neural network library
 *
 * ModelEvaluator.cpp
 */

#include "ModelEvaluator.hpp"
#include <cmath>
#include <algorithm>

namespace nn
{
    std::vector<double> ModelEvaluator::predict(const std::vector<double> &input)
    {
        // Set all BatchNormalization layers to inference mode
        setBatchTrainingMode(false);

        // Convert the input vector to a matrix
        Matrix output = Matrix(input.size(), 1, input);

        // Perform forward propagation
        output = forward(output);

        // Set all BatchNormalization layers back to training mode
        setBatchTrainingMode(true);

        // Return the output as a vector
        return output.getData();
    }

    std::vector<std::vector<double>> ModelEvaluator::predict(const std::vector<std::vector<double>> &input)
    {
        // Set all BatchNormalization layers to inference mode
        setBatchTrainingMode(false);

        // Perform forward propagation
        Matrix forwardOutput = forward(Matrix(input).transpose()).transpose();
        std::vector<std::vector<double>> result;

        // Convert the output matrix to a vector of vectors
        for (int i = 0; i < forwardOutput.getRows(); i++)
        {
            std::vector<double> row;
            for (int j = 0; j < forwardOutput.getCols(); j++)
                row.push_back(forwardOutput[{i, j}]);
            result.push_back(row);
        }

        // Set all BatchNormalization layers back to training mode
        setBatchTrainingMode(true);

        // Return the output vector
        return result;
    }

    double ModelEvaluator::evaluate(
        const std::vector<std::vector<double>> &xTest,
        const std::vector<std::vector<double>> &yTest,
        const e_metric metric
    )
    {
        // Check if the test data is empty
        if (xTest.empty() || yTest.empty())
            return 0.0;

        // Get predictions for the test data
        std::vector<std::vector<double>> predictions = predict(xTest);

        // Compute and return the specified metric
        return computeMetric(predictions, yTest, metric);
    }

    Matrix ModelEvaluator::forward(const Matrix &input)
    {
        // Propagate the input forward through the layers
        Matrix output = input;

        for (const auto &layer : m_layers)
            output = layer->forward(output);

        return output;
    }

    std::vector<double> ModelEvaluator::evaluate(
        const std::vector<std::vector<double>> &xTest,
        const std::vector<std::vector<double>> &yTest,
        const std::vector<e_metric> &metrics
    )
    {
        std::vector<double> result(2, 0.0);

        // Check if the test data is empty
        if (xTest.empty() || yTest.empty())
            return result;

        std::vector<std::vector<double>> predictions = predict(xTest);

        // Compute each metric
        for (auto const metric : metrics)
        {
            double temp = computeMetric(predictions, yTest, metric);
            result[static_cast<int>(metric)] = temp;
        }

        return result;
    }

    void ModelEvaluator::setBatchTrainingMode(const bool isTraining)
    {
        for (const auto &layer : m_layers)
        {
            if (layer->getType() == BATCH_NORM)
            {
                // Safely cast the Layer pointer to a BatchNormalization pointer
                BatchNormalization *bnLayer = dynamic_cast<BatchNormalization *>(layer.get());
                if (bnLayer)
                    bnLayer->setTrainingMode(isTraining);
            }
        }
    }

    double ModelEvaluator::computeMetric(
        const std::vector<std::vector<double>> &predictions,
        const std::vector<std::vector<double>> &targets,
        const e_metric metric
    )
    {
        switch (metric)
        {
        case ACCURACY_LOG:
            return computeAccuracy(predictions, targets);

        case MAE_LOG:
            return computeMAE(predictions, targets);
        }

        return 0.0;
    }

    double ModelEvaluator::computeAccuracy(
        const std::vector<std::vector<double>> &predictions,
        const std::vector<std::vector<double>> &targets
    )
    {
        int correct = 0;

        // Determine if the task is classification or regression
        bool isClassification = (predictions[0].size() > 1);

        for (int i = 0; i < predictions.size(); i++)
        {
            // If classification (multiple outputs)
            if (isClassification)
            {
                // Find the index of the maximum output
                int maxOutputIndex = std::max_element(predictions[i].begin(), predictions[i].end()) - predictions[i].begin();
                int maxTargetIndex = std::max_element(targets[i].begin(), targets[i].end()) - targets[i].begin();

                // Check if the prediction is correct
                if (maxOutputIndex == maxTargetIndex)
                    correct++;
            }
            // If regression (single output)
            else if (std::round(predictions[i][0]) == std::round(targets[i][0]))
                correct++;
        }

        // Return the accuracy
        return static_cast<double>(correct) / targets.size();
    }

    double ModelEvaluator::computeMAE(
        const std::vector<std::vector<double>> &predictions,
        const std::vector<std::vector<double>> &targets
    )
    {
        // Convert predictions and targets to matrices
        Matrix predictionsMatrix = Matrix(predictions);
        Matrix targetsMatrix = Matrix(targets);

        // Compute the absolute error and average it
        return (targetsMatrix - predictionsMatrix).map([](double x) { return std::abs(x); }).sum() / predictionsMatrix.getRows();
    }
}