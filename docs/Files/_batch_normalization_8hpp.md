# Layers/BatchNormalization/BatchNormalization.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::BatchNormalization](../Classes/classnn_1_1_batch_normalization.md)** <br>Implements Batch Normalization layer.  |




## Source code

```cpp


#ifndef BATCHNORMALIZATION_HPP
#define BATCHNORMALIZATION_HPP

#include "../Common/Layer.hpp"

namespace nn
{
    class BatchNormalization : public Layer
    {
    private:
        Matrix m_input;       
        Matrix m_gamma;       
        Matrix m_beta;        
        Matrix m_normalized;  
        Matrix m_gradGamma;   
        Matrix m_gradBeta;    
        Matrix m_mean;        
        Matrix m_stddev;      
        Matrix m_runningMean; 
        Matrix m_runningVar;  
        double m_epsilon;     
        double m_momentum;    
        bool m_isTraining;    

    public:
        BatchNormalization(const int numFeatures, const double momentum = 0.99, const double epsilon = 1e-15);

        BatchNormalization(std::ifstream &file);

        Matrix forward(const Matrix &input) override;

        Matrix backward(const Matrix &gradient) override;

        void resetGradients() override;

        void applyGradient(Optimizer &optimizer, const int batchSize) override;

        void save(std::ofstream &file) const override;

        e_layerType getType() const override { return BATCH_NORM; };

        void setTrainingMode(const bool isTrainging) { m_isTraining = isTrainging; };
    };
}

#endif
```
