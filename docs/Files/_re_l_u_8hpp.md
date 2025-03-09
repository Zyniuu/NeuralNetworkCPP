# Activations/ReLU/ReLU.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::ReLU](../Classes/classnn_1_1_re_l_u.md)** <br>Implements the Rectified Linear Unit ([ReLU]()) activation function.  |




## Source code

```cpp


#ifndef RELU_HPP
#define RELU_HPP

#include "../Common/Activation.hpp"

namespace nn
{
    class ReLU : public Activation
    {
    public:
        Matrix forward(const Matrix &input) override;

        Matrix backward(const Matrix &gradient) override;
    };
}

#endif
```
