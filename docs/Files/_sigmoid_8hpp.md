# Activations/Sigmoid/Sigmoid.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Sigmoid](../Classes/classnn_1_1_sigmoid.md)** <br>Implements the [Sigmoid]() activation function.  |




## Source code

```cpp


#ifndef SIGMOID_HPP
#define SIGMOID_HPP

#include "../Common/Activation.hpp"

namespace nn
{
    class Sigmoid : public Activation
    {
    public:
        Matrix forward(const Matrix &input) override;

        Matrix backward(const Matrix &gradient) override;
    };
}

#endif
```
