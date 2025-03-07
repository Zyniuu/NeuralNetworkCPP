# DataPreprocessing/Scalers/Common/Scaler.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Scaler](../Classes/classnn_1_1_scaler.md)** <br>Abstract base class for all scalers.  |




## Source code

```cpp


#ifndef SCALER_HPP
#define SCALER_HPP

#include <vector>

namespace nn
{
    class Scaler
    {
    public:
        virtual void fit(const std::vector<std::vector<double>> &data) = 0;

        virtual std::vector<std::vector<double>> transform(const std::vector<std::vector<double>> &data) = 0;

        virtual std::vector<std::vector<double>> fitTransform(const std::vector<std::vector<double>> &data) = 0;
    };
}

#endif
```
