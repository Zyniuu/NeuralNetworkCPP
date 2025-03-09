# Optimizers/SGD/SGD.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::SGD](../Classes/classnn_1_1_s_g_d.md)** <br>Stochastic Gradient Descent ([SGD]()) optimizer with momentum.  |




## Source code

```cpp


#ifndef SGD_HPP
#define SGD_HPP

#include "../Common/Optimizer.hpp"
#include <unordered_map>

namespace nn
{
    class SGD : public Optimizer
    {
    private:
        double m_momentum;                                 
        std::unordered_map<Matrix *, Matrix> m_velocities; 

    public:
        SGD(double learningRate = 0.001, double momentum = 0.9);

        void update(Matrix &weights, Matrix &biases, const Matrix &gradWeights, const Matrix &gradBiases) override;
    };
}

#endif
```
