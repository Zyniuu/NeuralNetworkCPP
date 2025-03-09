# DataPreprocessing/Scalers/MinMaxScaler/MinMaxScaler.cpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |




## Source code

```cpp


#include "MinMaxScaler.hpp"
#include <stdexcept>
#include <limits>

namespace nn
{
    MinMaxScaler::MinMaxScaler(const double featureRangeMin, const double featureRangeMax)
        : m_featureRangeMin(featureRangeMin), m_featureRangeMax(featureRangeMax) {}

    void MinMaxScaler::fit(const std::vector<std::vector<double>> &data)
    {
        if (data.empty() || data[0].empty())
            throw std::runtime_error("Input data is empty");

        int numFeatures = data[0].size();
        m_min.resize(numFeatures, std::numeric_limits<double>::max());
        m_max.resize(numFeatures, std::numeric_limits<double>::lowest());

        // Compute min and max for each feature
        for (const auto &row : data)
        {
            for (int i = 0; i < numFeatures; i++)
            {
                if (row[i] < m_min[i])
                    m_min[i] = row[i];
                if (row[i] > m_max[i])
                    m_max[i] = row[i];
            }
        }
    }

    std::vector<std::vector<double>> MinMaxScaler::transform(const std::vector<std::vector<double>> &data)
    {
        if (data.empty() || data[0].empty())
            throw std::runtime_error("Input data is empty");

        std::vector<std::vector<double>> normalizedData = data;

        for (auto &row : normalizedData)
        {
            for (int i = 0; i < row.size(); i++)
            {
                if (m_max[i] == m_min[i])
                    row[i] = m_featureRangeMin;
                else
                    row[i] = (row[i] - m_min[i]) / (m_max[i] - m_min[i]) * (m_featureRangeMax - m_featureRangeMin) + m_featureRangeMin;
            }
        }

        return normalizedData;
    }

    std::vector<std::vector<double>> MinMaxScaler::fitTransform(const std::vector<std::vector<double>> &data)
    {
        fit(data);
        return transform(data);
    }
}
```
