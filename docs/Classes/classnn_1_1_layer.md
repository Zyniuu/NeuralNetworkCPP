# nn::Layer



Abstract base class for neural network layers.  [More...](#detailed-description)


`#include <Layer.hpp>`

Inherited by [nn::BatchNormalization](classnn_1_1_batch_normalization.md), [nn::DenseLayer](classnn_1_1_dense_layer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| virtual [Matrix](classnn_1_1_matrix.md) | **[forward](classnn_1_1_layer.md#function-forward)**(const [Matrix](classnn_1_1_matrix.md) & input) =0<br>Performs forward propagation.  |
| virtual [Matrix](classnn_1_1_matrix.md) | **[backward](classnn_1_1_layer.md#function-backward)**(const [Matrix](classnn_1_1_matrix.md) & gradient, [Optimizer](classnn_1_1_optimizer.md) & optimizer) =0<br>Performs backward propagation.  |
| virtual void | **[save](classnn_1_1_layer.md#function-save)**(std::ofstream & file) const =0<br>Saves the layer's state to a binary file.  |
| virtual [e_layerType](../Namespaces/namespacenn.md#enum-e_layertype) | **[getType](classnn_1_1_layer.md#function-gettype)**() const =0<br>Returns the type of the layer.  |

## Detailed Description

```cpp
class nn::Layer;
```

Abstract base class for neural network layers. 

This class defines the interface for forward and backward propagation, as well as saving layer state.

## Public Functions Documentation

### function forward

```cpp
virtual Matrix forward(
    const Matrix & input
) =0
```

Performs forward propagation. 

**Parameters**: 

  * **input** The input matrix. 


**Return**: The output matrix after applying the layer's transformation. 

**Reimplemented by**: [nn::BatchNormalization::forward](classnn_1_1_batch_normalization.md#function-forward), [nn::DenseLayer::forward](classnn_1_1_dense_layer.md#function-forward)


### function backward

```cpp
virtual Matrix backward(
    const Matrix & gradient,
    Optimizer & optimizer
) =0
```

Performs backward propagation. 

**Parameters**: 

  * **gradient** The gradient of the loss with respect to the output.
  * **optimizer** The optimizer to use for weights and biases updates.


**Return**: The gradient of the loss with respect to the input. 

**Reimplemented by**: [nn::BatchNormalization::backward](classnn_1_1_batch_normalization.md#function-backward), [nn::DenseLayer::backward](classnn_1_1_dense_layer.md#function-backward)


### function save

```cpp
virtual void save(
    std::ofstream & file
) const =0
```

Saves the layer's state to a binary file. 

**Parameters**: 

  * **file** Output file stream (must be opened in binary mode). 


**Reimplemented by**: [nn::BatchNormalization::save](classnn_1_1_batch_normalization.md#function-save), [nn::DenseLayer::save](classnn_1_1_dense_layer.md#function-save)


### function getType

```cpp
virtual e_layerType getType() const =0
```

Returns the type of the layer. 

**Return**: The layer type as an enum value. 

**Reimplemented by**: [nn::BatchNormalization::getType](classnn_1_1_batch_normalization.md#function-gettype), [nn::DenseLayer::getType](classnn_1_1_dense_layer.md#function-gettype)
