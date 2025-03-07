# Initializers/XavierNormal/XavierNormal.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::XavierNormal](../Classes/classnn_1_1_xavier_normal.md)** <br>Implements Xavier (Glorot) normal initialization for neural network weights.  |




## Source code

```cpp


#ifndef XAVIERNORMAL_HPP
#define XAVIERNORMAL_HPP

#include "../Common/Initializer.hpp"

namespace nn
{
    class XavierNormal : public Initializer
    {
    private:
        std::normal_distribution<double> m_dist; 

    public:
        XavierNormal(const int inputs, const int outputs);

        double getRandomNum() override;
    };
}

#endif
```
