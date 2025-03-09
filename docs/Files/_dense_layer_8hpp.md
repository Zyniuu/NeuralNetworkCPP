# Layers/DenseLayer/DenseLayer.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::DenseLayer](../Classes/classnn_1_1_dense_layer.md)** <br>Implements a fully connected (dense) layer.  |




## Source code

```cpp


#ifndef DENSELAYER_HPP
#define DENSELAYER_HPP

#include "../Common/Layer.hpp"
#include "../../Activations/Common/Activation.hpp"
#include <memory>

namespace nn
{
    class DenseLayer : public Layer
    {
    private:
        Matrix m_weights;                         
        Matrix m_biases;                          
        Matrix m_input;                           
        Matrix m_gradWeights;                     
        Matrix m_gradBiases;                      
        Matrix m_output;                          
        std::unique_ptr<Activation> m_activation; 
        e_activation m_activationID;              

    public:
        DenseLayer(const int inputSize, const int outputSize, e_initializer initializerID, e_activation activationID);

        DenseLayer(std::ifstream &file);

        Matrix forward(const Matrix &input) override;

        Matrix backward(const Matrix &gradient) override;

        void resetGradients() override;

        void applyGradient(Optimizer &optimizer, const int batchSize) override;

        void save(std::ofstream &file) const override;

        e_layerType getType() const override { return DENSE; }

    private:
        void initWeights(const int inputSize, const int outputSize, e_initializer initializerID);

        void initActivationFunction(e_activation activationID);
    };
}

#endif
```
