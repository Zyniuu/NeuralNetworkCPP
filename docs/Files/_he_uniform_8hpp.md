# Initializers/HeUniform/HeUniform.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::HeUniform](../Classes/classnn_1_1_he_uniform.md)** <br>Implements He Uniform initialization for neural network weights.  |




## Source code

```cpp


#ifndef HEUNIFORM_HPP
#define HEUNIFORM_HPP

#include "../Common/Initializer.hpp"

namespace nn
{
    class HeUniform : public Initializer
    {
    private:
        std::uniform_real_distribution<double> m_dist; 

    public:
        HeUniform(const int inputs, const int outputs);

        double getRandomNum() override;
    };
}

#endif
```
