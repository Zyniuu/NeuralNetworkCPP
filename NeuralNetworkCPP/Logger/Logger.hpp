/**
 * C++ neural network library
 *
 * Logger.hpp
 */

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <vector>
#include <chrono>

namespace nn
{
    /**
     * @brief Enum with avaible metrics.
     */
    enum e_metric { ACCURACY_LOG, MAE_LOG };

    /**
     * @class Logger
     * @brief Handles logging of training progress and metrics.
     */
    class Logger
    {
    private:
        int m_progressBarLength; ///< Max length of the progress bar.
        std::chrono::time_point<std::chrono::steady_clock> m_trainStart; ///< Start time of training.
        std::chrono::time_point<std::chrono::steady_clock> m_epochStart; ///< Start time of a batch.

    public:
        /**
         * @brief Constructs a Logger with the specified metrics.
         *
         * @param progressBarLength Max length of the progress bar.
         */
        Logger(const int progressBarLength = 30);

        /**
         * @brief Starts the training timer.
         */
        void logTrainingStart();

        /**
         * @brief Stops the training timer and logs the end of training.
         * 
         * @param isEarlyStopped Flag for displaying end of training or early stopped message.
         */
        void logTrainingEnd(const bool isEarlyStopped);

        /**
         * @brief Logs the start of an epoch.
         *
         * @param currentEpoch Current epoch number.
         * @param totalEpochs Total number of epochs.
         */
        void logEpochStart(const int currentEpoch, const int totalEpochs);

        /**
         * @brief Logs the end of an epoch.
         *
         * @param totalBatches Total number of baches.
         * @param loss Computed loss between predictions and targets.
         * @param computedMetrics Computed metrics on validation set.
         * @param metrics Metrics to log.
         */
        void logEpochEnd(
            const int totalBatches,
            const double loss,
            const std::vector<double> &computedMetrics,
            const std::vector<e_metric> &metrics
        );

        /**
         * @brief Logs the progress of a batch.
         *
         * @param currentBatch Current batch number.
         * @param totalBatches Total number of batches.
         */
        void logBatch(const int currentBatch, const int totalBatches);
    
    private:
        /**
         * @brief Logs the metric info.
         *
         * @param metric The metric to log.
         * @param computedMetrics Computed metrics on validation set.
         */
        void logMetric(const e_metric metric, const std::vector<double> &computedMetrics);
    };
}

#endif