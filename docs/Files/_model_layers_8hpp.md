# ModelParts/ModelLayers/ModelLayers.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::ModelLayers](../Classes/classnn_1_1_model_layers.md)** <br>Manages the layers of a neural network model.  |




## Source code

```cpp


#include "../../Layers/Layers.hpp"

namespace nn
{
    class ModelLayers
    {
    protected:
        std::vector<std::unique_ptr<Layer>> m_layers; 

    public:
        void addLayer(std::unique_ptr<Layer> layer);

    protected:
        void initLayer(e_layerType layerType, std::ifstream &file);
    };
}
```
