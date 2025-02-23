/**
 * C++ neural network library
 *
 * ModelEvaluator.hpp
 */

#include "../ModelLayers/ModelLayers.hpp"
#include "../../Logger/Logger.hpp"

namespace nn
{
    /**
     * @class ModelEvaluator
     * @brief Handles model evaluation, prediction, and metric computation.
     *
     * This class extends ModelLayers and provides functionality for making predictions,
     * evaluating the model, and computing metrics like accuracy and mean absolute error.
     */
    class ModelEvaluator : public ModelLayers
    {
    public:
        /**
         * @brief Predicts the output for a given input.
         *
         * @param input The input vector.
         * @return The predicted output vector.
         */
        std::vector<double> predict(const std::vector<double> &input);

        /**
         * @brief Predicts the output for a given vector of inputs.
         *
         * @param input The vector of vector of inputs.
         * @return The predicted vector of vector of outputs.
         */
        std::vector<std::vector<double>> predict(const std::vector<std::vector<double>> &input);

        /**
         * @brief Evaluates the model on the provided test data.
         *
         * @param xTest Test data (vector of input vectors).
         * @param yTest Test labels (vector of output vectors).
         * @param metric The metric to compute (default: ACCURACY_LOG).
         * @return The computed metric.
         */
        double evaluate(
            const std::vector<std::vector<double>> &xTest,
            const std::vector<std::vector<double>> &yTest,
            const e_metric metric = ACCURACY_LOG
        );

    protected:
        /**
         * @brief Performs forward propagation through the network.
         *
         * @param input The input matrix.
         * @return The output matrix.
         */
        Matrix forward(const Matrix &input);

        /**
         * @brief Evaluates the model on the provided test data.
         *
         * @param xTest Test data (vector of input vectors).
         * @param yTest Test labels (vector of output vectors).
         * @param metrics The vector of metrics to compute.
         * @return The vector of computed metrics.
         */
        std::vector<double> evaluate(
            const std::vector<std::vector<double>> &xTest,
            const std::vector<std::vector<double>> &yTest,
            const std::vector<e_metric> &metrics
        );

    private:
        /**
         * @brief Set the training flag for all BatchNormalization layers.
         *
         * @param isTraining Flag to set for BatchNormalization layers.
         */
        void setBatchTrainingMode(const bool isTraining);

        /**
         * @brief Computes the provided metric.
         *
         * @param predictions Vector of vectors of doubles of predictions.
         * @param targets Vector of vectors of doubles of expected targets.
         * @param metric Metric to compute.
         * @return Computed metric based on provided predictions and targets.
         */
        double computeMetric(
            const std::vector<std::vector<double>> &predictions,
            const std::vector<std::vector<double>> &targets,
            const e_metric metric
        );

        /**
         * @brief Computes the model's accuracy.
         *
         * @param predictions Vector of vectors of doubles of predictions.
         * @param targets Vector of vectors of doubles of expected targets.
         * @return Computed accuracy based on provided predictions and targets.
         */
        double computeAccuracy(
            const std::vector<std::vector<double>> &predictions,
            const std::vector<std::vector<double>> &targets
        );

        /**
         * @brief Computes the model's Mean Absolute Error.
         *
         * @param predictions Vector of vectors of doubles of predictions.
         * @param targets Vector of vectors of doubles of expected targets.
         * @return Computed MAE based on provided predictions and targets.
         */
        double computeMAE(
            const std::vector<std::vector<double>> &predictions,
            const std::vector<std::vector<double>> &targets
        );
    };
}