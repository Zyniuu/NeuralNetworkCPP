# Initializers/Common/Initializer.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Initializer](../Classes/classnn_1_1_initializer.md)** <br>Abstract base class for weight initializers in neural networks.  |




## Source code

```cpp


#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

#include <random>

namespace nn
{
    class Initializer
    {
    protected:
        int m_inputs;       
        int m_outputs;      
        std::mt19937 m_gen; 

    public:
        Initializer(const int inputs, const int outputs)
            : m_inputs(inputs), m_outputs(outputs), m_gen(std::random_device{}()) {}

        virtual double getRandomNum() = 0;
    };
}

#endif
```
