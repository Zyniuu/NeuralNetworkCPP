# Layers/Common/Layer.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Layer](../Classes/classnn_1_1_layer.md)** <br>Abstract base class for neural network layers.  |




## Source code

```cpp


#ifndef LAYER_HPP
#define LAYER_HPP

#include "../../Matrix/Matrix.hpp"
#include "../../Optimizers/Common/Optimizer.hpp"
#include <fstream>

namespace nn
{
    enum e_layerType { DENSE, BATCH_NORM };

    enum e_initializer { HE_NORMAL, HE_UNIFORM, XAVIER_NORMAL, XAVIER_UNIFORM };

    enum e_activation { RELU, SIGMOID, SOFTMAX, NONE };

    class Layer
    {
    public:
        virtual Matrix forward(const Matrix &input) = 0;

        virtual Matrix backward(const Matrix &gradient) = 0;

        virtual void resetGradients() = 0;

        virtual void applyGradient(Optimizer &optimizer, const int batchSize) = 0;

        virtual void save(std::ofstream &file) const = 0;

        virtual e_layerType getType() const = 0;
    };
}

#endif
```
