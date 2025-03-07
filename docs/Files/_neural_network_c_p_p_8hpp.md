# NeuralNetworkCPP.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::NeuralNetworkCPP](../Classes/classnn_1_1_neural_network_c_p_p.md)** <br>A neural network model that supports adding layers, compiling, training, and saving/loading.  |




## Source code

```cpp


#ifndef NEURALNETWORKCPP_HPP
#define NEURALNETWORKCPP_HPP

#include "ModelParts/ModelTrainer/ModelTrainer.hpp"
#include <thread>

namespace nn
{
    class NeuralNetworkCPP : public ModelTrainer
    {
    public:
        NeuralNetworkCPP(const int numThreads = std::thread::hardware_concurrency());

        NeuralNetworkCPP(const std::string &filename, const int numThreads = std::thread::hardware_concurrency());

        void save(const std::string &filename) const;
    };
}

#endif
```
