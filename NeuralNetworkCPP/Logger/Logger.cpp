/**
 * C++ neural network library
 *
 * Logger.cpp
 */

#include "Logger.hpp"
#include <iostream>
#include <iomanip>
#include <string>

namespace nn
{
    Logger::Logger(const int progressBarLength)
        : m_progressBarLength(progressBarLength) {}

    void Logger::logTrainingStart()
    {
        // Start training timer
        std::cout << "[    START ] Training started." << std::endl << std::endl; 
        m_trainStart = std::chrono::steady_clock::now();
    }

    void Logger::logTrainingEnd(const bool isEarlyStopped)
    {
        // Get the total training duration
        auto duration = std::chrono::steady_clock::now() - m_trainStart;

        // Convert the duration to hours, minutes, seconds, and milliseconds
        auto hours = std::chrono::duration_cast<std::chrono::hours>(duration).count();
        duration -= std::chrono::hours(hours); // Subtract hours to get the remaining time

        auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration).count();
        duration -= std::chrono::minutes(minutes); // Subtract minutes to get the remaining time

        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
        duration -= std::chrono::seconds(seconds); // Subtract seconds to get the remaining time

        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        // Print the total training time
        if (isEarlyStopped)
            std::cout << "\n[  STOPPED ] Training stopped early: patience limit reached." << std::endl;
        else
            std::cout << "\n[     DONE ] Training done." << std::endl;
        std::cout << "[     TIME ] ";
        std::cout << hours << " hours ";
        std::cout << minutes << " minutes ";
        std::cout << seconds << " seconds ";
        std::cout << milliseconds << " milliseconds" << std::endl;
    }

    void Logger::logEpochStart(const int currentEpoch, const int totalEpochs)
    {
        // Start the epoch timer
        m_epochStart = std::chrono::steady_clock::now();

        // Print the epochs progress
        std::cout << "Epoch " << currentEpoch << "/" << totalEpochs << std::endl;
    }

    void Logger::logEpochEnd(
        const int totalBatches,
        const double loss,
        const std::vector<double> &computedMetrics,
        const std::vector<e_metric> &metrics
    )
    {
        // Convert epoch duration on seconds and milliseconds
        auto duration = std::chrono::steady_clock::now() - m_epochStart;
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        // Print epoch duration in seconds and step duration in milliseconds
        std::cout << " - " << seconds << "s " << (milliseconds / totalBatches) << "ms/step";
        std::cout << " - loss: " << std::setw(6) << loss;

        // Process the metrics
        for (const auto &metric : metrics)
            logMetric(metric, computedMetrics);

        std::cout << std::endl;
    }

    void Logger::logBatch(const int currentBatch, const int totalBatches)
    {
        // Determine how much to fill the progress bar
        int progress = (static_cast<double>(currentBatch) / static_cast<double>(totalBatches)) * static_cast<double>(m_progressBarLength);

        // Print the current batch number
        std::cout << '\r' << currentBatch << "/" << totalBatches;

        // Print the progress bar
        std::cout << " [" << std::string(progress, '=') << std::string(m_progressBarLength - progress, ' ') << "]";
    }

    void Logger::logMetric(const e_metric metric, const std::vector<double> &computedMetrics)
    {
        // Print the logs depending on selected metric
        switch (metric)
        {
        case ACCURACY_LOG:
            std::cout << " - accuracy: " << std::setw(6) << computedMetrics[static_cast<int>(ACCURACY_LOG)];
            break;

        case MAE_LOG:
            std::cout << " - mae: " << std::setw(6) << computedMetrics[static_cast<int>(MAE_LOG)];
            break;
        }
    }
}