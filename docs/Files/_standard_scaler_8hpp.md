# DataPreprocessing/Scalers/StandardScaler/StandardScaler.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::StandardScaler](../Classes/classnn_1_1_standard_scaler.md)** <br>Normalizes data to have a mean of 0 and a standard deviation of 1.  |




## Source code

```cpp


#ifndef STANDARDSCALER_HPP
#define STANDARDSCALER_HPP

#include "../Common/Scaler.hpp"

namespace nn
{
    class StandardScaler : public Scaler
    {
    private:
        std::vector<double> m_mean;   
        std::vector<double> m_stddev; 

    public:
        void fit(const std::vector<std::vector<double>> &data) override;

        std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> &data) override;

        std::vector<std::vector<double>> fitTransform(const std::vector<std::vector<double>> &data) override;
    };
}

#endif
```
