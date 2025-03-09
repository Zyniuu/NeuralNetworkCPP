# Losses/CategoricalCrossEntropy/CategoricalCrossEntropy.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::CategoricalCrossEntropy](../Classes/classnn_1_1_categorical_cross_entropy.md)** <br>Implements the Categorical Cross-Entropy loss function.  |




## Source code

```cpp


#ifndef CATEGORICALCROSSENTROPY_HPP
#define CATEGORICALCROSSENTROPY_HPP

#include "../Common/Loss.hpp"

namespace nn
{
    class CategoricalCrossEntropy : public Loss
    {
    public:
        double computeLoss(const Matrix &predictions, const Matrix &targets) override;

        Matrix computeGradient(const Matrix &predictions, const Matrix &targets) override;
    };
}

#endif
```
