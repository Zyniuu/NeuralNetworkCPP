# Initializers/XavierUniform/XavierUniform.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::XavierUniform](../Classes/classnn_1_1_xavier_uniform.md)** <br>Implements Xavier (Glorot) uniform initialization for neural network weights.  |




## Source code

```cpp


#ifndef XAVIERUNIFORM_HPP
#define XAVIERUNIFORM_HPP

#include "../Common/Initializer.hpp"

namespace nn
{
    class XavierUniform : public Initializer
    {
    private:
        std::uniform_real_distribution<double> m_dist; 

    public:
        XavierUniform(const int inputs, const int outputs);

        double getRandomNum() override;
    };
}

#endif
```
