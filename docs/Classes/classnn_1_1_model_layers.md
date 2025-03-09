# nn::ModelLayers



Manages the layers of a neural network model.  [More...](#detailed-description)


`#include <ModelLayers.hpp>`

Inherited by [nn::ModelEvaluator](classnn_1_1_model_evaluator.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| void | **[addLayer](classnn_1_1_model_layers.md#function-addlayer)**(std::unique_ptr< [Layer](classnn_1_1_layer.md) > layer)<br>Adds a layer to the network.  |

## Protected Functions

|                | Name           |
| -------------- | -------------- |
| void | **[initLayer](classnn_1_1_model_layers.md#function-initlayer)**([e_layerType](../Namespaces/namespacenn.md#enum-e_layertype) layerType, std::ifstream & file)<br>Initializes a layer based on the provided layer type.  |

## Protected Attributes

|                | Name           |
| -------------- | -------------- |
| std::vector< std::unique_ptr< [Layer](classnn_1_1_layer.md) > > | **[m_layers](classnn_1_1_model_layers.md#variable-m_layers)** <br>Vector of layers in the network.  |

## Detailed Description

```cpp
class nn::ModelLayers;
```

Manages the layers of a neural network model. 

This class provides functionality to add and initialize layers in a neural network. 

## Public Functions Documentation

### function addLayer

```cpp
void addLayer(
    std::unique_ptr< Layer > layer
)
```

Adds a layer to the network. 

**Parameters**: 

  * **layer** The layer to add. 


## Protected Functions Documentation

### function initLayer

```cpp
void initLayer(
    e_layerType layerType,
    std::ifstream & file
)
```

Initializes a layer based on the provided layer type. 

**Parameters**: 

  * **layerType** Enum value of the layer type. 
  * **file** Input file stream (must be opened in binary mode). 


**Exceptions**: 

  * **std::runtime_error** If the layer type is invalid. 


## Protected Attributes Documentation

### variable m_layers

```cpp
std::vector< std::unique_ptr< Layer > > m_layers;
```

Vector of layers in the network. 
