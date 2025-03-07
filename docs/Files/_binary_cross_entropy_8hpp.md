# Losses/BinaryCrossEntropy/BinaryCrossEntropy.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::BinaryCrossEntropy](../Classes/classnn_1_1_binary_cross_entropy.md)** <br>Implements the Binary Cross-Entropy loss function.  |




## Source code

```cpp


#ifndef BINARYCROSSENTROPY_HPP
#define BINARYCROSSENTROPY_HPP

#include "../Common/Loss.hpp"

namespace nn
{
    class BinaryCrossEntropy : public Loss
    {
    public:
        double computeLoss(const Matrix &predictions, const Matrix &targets) override;

        Matrix computeGradient(const Matrix &predictions, const Matrix &targets) override;
    };
}

#endif
```
