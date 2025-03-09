# Losses/Common/Loss.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Loss](../Classes/classnn_1_1_loss.md)** <br>Abstract base class for loss functions.  |




## Source code

```cpp


#ifndef LOSS_HPP
#define LOSS_HPP

#include "../../Matrix/Matrix.hpp"

namespace nn
{
    class Loss
    {
    protected:
        double m_epsilon = 1e-15; 

    public:
        virtual double computeLoss(const Matrix &predictions, const Matrix &targets) = 0;

        virtual Matrix computeGradient(const Matrix &predictions, const Matrix &targets) = 0;
    };
}

#endif
```
