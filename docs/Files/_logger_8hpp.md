# Logger/Logger.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Logger](../Classes/classnn_1_1_logger.md)** <br>Handles logging of training progress and metrics.  |




## Source code

```cpp


#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <vector>
#include <chrono>

namespace nn
{
    enum e_metric { ACCURACY_LOG, MAE_LOG };

    class Logger
    {
    private:
        int m_progressBarLength; 
        std::chrono::time_point<std::chrono::steady_clock> m_trainStart; 
        std::chrono::time_point<std::chrono::steady_clock> m_epochStart; 

    public:
        Logger(const int progressBarLength = 30);

        void logTrainingStart();

        void logTrainingEnd(const bool isEarlyStopped);

        void logEpochStart(const int currentEpoch, const int totalEpochs);

        void logEpochEnd(
            const int totalBatches,
            const double loss,
            const std::vector<double> &computedMetrics,
            const std::vector<e_metric> &metrics
        );

        void logBatch(const int currentBatch, const int totalBatches);
    
    private:
        void logMetric(const e_metric metric, const std::vector<double> &computedMetrics);
    };
}

#endif
```
