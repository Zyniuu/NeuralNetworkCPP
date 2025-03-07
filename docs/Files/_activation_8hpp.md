# Activations/Common/Activation.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Activation](../Classes/classnn_1_1_activation.md)** <br>Abstract base class for activation functions.  |




## Source code

```cpp


#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include "../../Matrix/Matrix.hpp"

namespace nn
{
    class Activation
    {
    protected:
        Matrix m_output; 

    public:
        virtual Matrix forward(const Matrix &input) = 0;

        virtual Matrix backward(const Matrix &gradient) = 0;
    };
}

#endif
```
