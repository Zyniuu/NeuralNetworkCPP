# Optimizers/RMSprop/RMSprop.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::RMSprop](../Classes/classnn_1_1_r_m_sprop.md)** <br>[RMSprop]() optimizer.  |




## Source code

```cpp


#ifndef RMSPROP_HPP
#define RMSPROP_HPP

#include "../Common/Optimizer.hpp"
#include <unordered_map>

namespace nn
{
    class RMSprop : public Optimizer
    {
    private:
        double m_gamma;                           
        double m_epsilon;                         
        std::unordered_map<Matrix *, Matrix> m_v; 

    public:
        RMSprop(double learningRate = 0.001, double gamma = 0.9, double epsilon = 1e-8);

        void update(Matrix &weights, Matrix &biases, const Matrix &gradWeights, const Matrix &gradBiases) override;
    };
}

#endif
```
