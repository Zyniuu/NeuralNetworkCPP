# Losses/MeanSquaredError/MeanSquaredError.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::MeanSquaredError](../Classes/classnn_1_1_mean_squared_error.md)** <br>Implements the Mean Squared Error (MSE) loss function.  |




## Source code

```cpp


#ifndef MEANSQUAREDERROR_HPP
#define MEANSQUAREDERROR_HPP

#include "../Common/Loss.hpp"

namespace nn
{
    class MeanSquaredError : public Loss
    {
    public:
        double computeLoss(const Matrix &predictions, const Matrix &targets) override;

        Matrix computeGradient(const Matrix &predictions, const Matrix &targets) override;
    };
}

#endif
```
