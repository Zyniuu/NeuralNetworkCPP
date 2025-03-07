# Optimizers/Common/Optimizer.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Optimizer](../Classes/classnn_1_1_optimizer.md)** <br>Abstract base class for optimizers.  |




## Source code

```cpp


#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "../../Matrix/Matrix.hpp"

namespace nn
{
    class Optimizer
    {
    protected:
        double m_learningRate; 

    public:
        Optimizer(double learningRate) : m_learningRate(learningRate) {}

        virtual void update(Matrix &weights, Matrix &biases, const Matrix &gradWeights, const Matrix &gradBiases) = 0;
    };
}

#endif
```
