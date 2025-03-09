# Initializers/HeNormal/HeNormal.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::HeNormal](../Classes/classnn_1_1_he_normal.md)** <br>Implements He Normal initialization for neural network weights.  |




## Source code

```cpp


#ifndef HENORMAL_HPP
#define HENORMAL_HPP

#include "../Common/Initializer.hpp"

namespace nn
{
    class HeNormal : public Initializer
    {
    private:
        std::normal_distribution<double> m_dist; 

    public:
        HeNormal(const int inputs, const int outputs);

        double getRandomNum() override;
    };
}

#endif
```
