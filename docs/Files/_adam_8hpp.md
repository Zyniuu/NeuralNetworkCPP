# Optimizers/Adam/Adam.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Adam](../Classes/classnn_1_1_adam.md)** <br>[Adam]() optimizer.  |




## Source code

```cpp


#ifndef ADAM_HPP
#define ADAM_HPP

#include "../Common/Optimizer.hpp"
#include <unordered_map>

namespace nn
{
    class Adam : public Optimizer
    {
    private:
        double m_beta1;   
        double m_beta2;   
        double m_epsilon; 

        // Maps for first and second moment estimates
        std::unordered_map<Matrix *, Matrix> m_m; 
        std::unordered_map<Matrix *, Matrix> m_v; 
        std::unordered_map<Matrix *, int> m_t;    

    public:
        Adam(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

        void update(Matrix &weights, Matrix &biases, const Matrix &gradWeights, const Matrix &gradBiases) override;
    };
}

#endif
```
