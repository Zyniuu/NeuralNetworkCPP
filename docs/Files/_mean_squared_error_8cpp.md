# Losses/MeanSquaredError/MeanSquaredError.cpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |




## Source code

```cpp


#include "MeanSquaredError.hpp"
#include <numeric>

namespace nn
{
    double MeanSquaredError::computeLoss(const Matrix &predictions, const Matrix &targets)
    {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols())
            throw std::invalid_argument("Predictions and targets must have the same dimensions.");

        Matrix error = (targets - predictions).map([](double x) { 
            return x * x; 
        });

        return error.sum() / error.getCols();
    }

    Matrix MeanSquaredError::computeGradient(const Matrix &predictions, const Matrix &targets)
    {
        if (predictions.getRows() != targets.getRows() || predictions.getCols() != targets.getCols())
            throw std::invalid_argument("Predictions and targets must have the same dimensions.");

        double scale = 2.0 / (predictions.getRows() * predictions.getCols());
        return scale * (predictions - targets);
    }
}
```
