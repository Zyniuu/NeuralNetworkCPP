# DataPreprocessing/Scalers/StandardScaler/StandardScaler.cpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |




## Source code

```cpp


#include "StandardScaler.hpp"
#include <stdexcept>
#include <cmath>

namespace nn
{
    void StandardScaler::fit(const std::vector<std::vector<double>> &data)
    {
        if (data.empty() || data[0].empty())
            throw std::runtime_error("Input data is empty");
        
        int numFeatures = data[0].size();
        m_mean.resize(numFeatures, 0.0);
        m_stddev.resize(numFeatures, 0.0);

        // Compute mean
        for (const auto &row : data)
            for (int i = 0; i < numFeatures; i++)
                m_mean[i] += row[i];
        
        for (int i = 0; i < numFeatures; i++)
            m_mean[i] /= data.size();
        
        // Compute standard deviation
        for (const auto &row : data)
            for (int i = 0; i < numFeatures; i++)
                m_stddev[i] += std::pow(row[i] - m_mean[i], 2);

        for (int i = 0; i < numFeatures; i++)
            m_stddev[i] = std::sqrt(m_stddev[i] / data.size());
    }

    std::vector<std::vector<double>> StandardScaler::transform(const std::vector<std::vector<double>> &data)
    {
        if (data.empty() || data[0].empty())
            throw std::runtime_error("Input data is empty");
        
        std::vector<std::vector<double>> normalizedData = data;

        for (auto &row : normalizedData)
            for (int i = 0; i < row.size(); i++)
                row[i] = (row[i] - m_mean[i]) / m_stddev[i];
            
        return normalizedData;
    }

    std::vector<std::vector<double>> StandardScaler::fitTransform(const std::vector<std::vector<double>> &data)
    {
        fit(data);
        return transform(data);
    }
}
```
