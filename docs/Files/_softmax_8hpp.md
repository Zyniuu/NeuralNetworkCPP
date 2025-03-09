# Activations/Softmax/Softmax.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Softmax](../Classes/classnn_1_1_softmax.md)** <br>Implements the [Softmax]() activation function.  |




## Source code

```cpp


#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "../Common/Activation.hpp"

namespace nn
{
    class Softmax : public Activation
    {
    public:
        Matrix forward(const Matrix &input) override;

        Matrix backward(const Matrix &gradient) override;
    };
}

#endif
```
