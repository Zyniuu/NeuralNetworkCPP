# DataPreprocessing/Scalers/MinMaxScaler/MinMaxScaler.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::MinMaxScaler](../Classes/classnn_1_1_min_max_scaler.md)** <br>Normalizes data to a specified range (default: [0, 1]).  |




## Source code

```cpp


#ifndef MINMAXSCALER_HPP
#define MINMAXSCALER_HPP

#include "../Common/Scaler.hpp"

namespace nn
{
    class MinMaxScaler : public Scaler
    {
    private:
        std::vector<double> m_min; 
        std::vector<double> m_max; 
        double m_featureRangeMin;  
        double m_featureRangeMax;  

    public:
        MinMaxScaler(const double featureRangeMin = 0.0, const double featureRangeMax = 1.0);

        void fit(const std::vector<std::vector<double>> &data) override;

        std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> &data) override;

        std::vector<std::vector<double>> fitTransform(const std::vector<std::vector<double>> &data) override;
    };
}

#endif
```
