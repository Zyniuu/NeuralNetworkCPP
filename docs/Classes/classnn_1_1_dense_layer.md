# nn::DenseLayer



Implements a fully connected (dense) layer.  [More...](#detailed-description)


`#include <DenseLayer.hpp>`

Inherits from [nn::Layer](classnn_1_1_layer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[DenseLayer](classnn_1_1_dense_layer.md#function-denselayer)**(const int inputSize, const int outputSize, [e_initializer](../Namespaces/namespacenn.md#enum-e_initializer) initializerID, [e_activation](../Namespaces/namespacenn.md#enum-e_activation) activationID)<br>Constructs a dense layer.  |
| | **[DenseLayer](classnn_1_1_dense_layer.md#function-denselayer)**(std::ifstream & file)<br>Constructs a dense layer from the file.  |
| virtual [Matrix](classnn_1_1_matrix.md) | **[forward](classnn_1_1_dense_layer.md#function-forward)**(const [Matrix](classnn_1_1_matrix.md) & input) override<br>Performs forward propagation.  |
| virtual [Matrix](classnn_1_1_matrix.md) | **[backward](classnn_1_1_dense_layer.md#function-backward)**(const [Matrix](classnn_1_1_matrix.md) & gradient) override<br>Performs backward propagation.  |
| virtual void | **[resetGradients](classnn_1_1_dense_layer.md#function-resetgradients)**() override<br>Resets the accumulated gradients of the layer.  |
| virtual void | **[applyGradient](classnn_1_1_dense_layer.md#function-applygradient)**([Optimizer](classnn_1_1_optimizer.md) & optimizer, const int batchSize) override<br>Applies accumulated gradients to the layer.  |
| virtual void | **[save](classnn_1_1_dense_layer.md#function-save)**(std::ofstream & file) const override<br>Saves the layer's state to a binary file.  |
| virtual [e_layerType](../Namespaces/namespacenn.md#enum-e_layertype) | **[getType](classnn_1_1_dense_layer.md#function-gettype)**() const override<br>Returns the type of the layer.  |

## Detailed Description

```cpp
class nn::DenseLayer;
```

Implements a fully connected (dense) layer. 

This layer applies a linear transformation (weights * input + biases) followed by an optional activation function. It supports saving and loading layer state to/from files. 

## Public Functions Documentation

### function DenseLayer

```cpp
DenseLayer(
    const int inputSize,
    const int outputSize,
    e_initializer initializerID,
    e_activation activationID
)
```

Constructs a dense layer. 

**Parameters**: 

  * **inputSize** Number of input neurons. 
  * **outputSize** Number of output neurons. 
  * **initializerID** Weights initializer. 
  * **activationID** Optional activation function. 


### function DenseLayer

```cpp
DenseLayer(
    std::ifstream & file
)
```

Constructs a dense layer from the file. 

**Parameters**: 

  * **file** Input file stream (must be opened in binary mode). 


**Exceptions**: 

  * **std::runtime_error** If the file is not open or reading fails. 


### function forward

```cpp
virtual Matrix forward(
    const Matrix & input
) override
```

Performs forward propagation. 

**Parameters**: 

  * **input** The input matrix. 


**Return**: The output matrix after applying the layer's transformation. 

**Reimplements**: [nn::Layer::forward](classnn_1_1_layer.md#function-forward)


### function backward

```cpp
virtual Matrix backward(
    const Matrix & gradient
) override
```

Performs backward propagation. 

**Parameters**: 

  * **gradient** The gradient of the loss with respect to the output. 


**Return**: The gradient of the loss with respect to the input. 

**Reimplements**: [nn::Layer::backward](classnn_1_1_layer.md#function-backward)


### function resetGradients

```cpp
virtual void resetGradients() override
```

Resets the accumulated gradients of the layer. 

**Reimplements**: [nn::Layer::resetGradients](classnn_1_1_layer.md#function-resetgradients)


### function applyGradient

```cpp
virtual void applyGradient(
    Optimizer & optimizer,
    const int batchSize
) override
```

Applies accumulated gradients to the layer. 

**Parameters**: 

  * **optimizer** The optimizer to use for weights and biases updates. 
  * **batchSize** Size of the batch from witch gradients were accumulated. 


**Reimplements**: [nn::Layer::applyGradient](classnn_1_1_layer.md#function-applygradient)


### function save

```cpp
virtual void save(
    std::ofstream & file
) const override
```

Saves the layer's state to a binary file. 

**Parameters**: 

  * **file** Output file stream (must be opened in binary mode). 


**Exceptions**: 

  * **std::runtime_error** If the file is not open or reading fails. 


**Reimplements**: [nn::Layer::save](classnn_1_1_layer.md#function-save)


### function getType

```cpp
inline virtual e_layerType getType() const override
```

Returns the type of the layer. 

**Return**: The layer type as an enum value. 

**Reimplements**: [nn::Layer::getType](classnn_1_1_layer.md#function-gettype)
