# nn::BatchNormalization



Implements Batch Normalization layer.  [More...](#detailed-description)


`#include <BatchNormalization.hpp>`

Inherits from [nn::Layer](classnn_1_1_layer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[BatchNormalization](classnn_1_1_batch_normalization.md#function-batchnormalization)**(const int numFeatures, const double momentum =0.99, const double epsilon =1e-15)<br>Constructs a [BatchNormalization](classnn_1_1_batch_normalization.md) layer.  |
| | **[BatchNormalization](classnn_1_1_batch_normalization.md#function-batchnormalization)**(std::ifstream & file)<br>Constructs a [BatchNormalization](classnn_1_1_batch_normalization.md) layer from a file.  |
| virtual [Matrix](classnn_1_1_matrix.md) | **[forward](classnn_1_1_batch_normalization.md#function-forward)**(const [Matrix](classnn_1_1_matrix.md) & input) override<br>Performs forward propagation.  |
| virtual [Matrix](classnn_1_1_matrix.md) | **[backward](classnn_1_1_batch_normalization.md#function-backward)**(const [Matrix](classnn_1_1_matrix.md) & gradient, [Optimizer](classnn_1_1_optimizer.md) & optimizer) override<br>Performs backward propagation.  |
| virtual void | **[save](classnn_1_1_batch_normalization.md#function-save)**(std::ofstream & file) const override<br>Saves the layer's state to a binary file.  |
| virtual [e_layerType](../Namespaces/namespacenn.md#enum-e_layertype) | **[getType](classnn_1_1_batch_normalization.md#function-gettype)**() const override<br>Returns the type of the layer.  |
| void | **[setTrainingMode](classnn_1_1_batch_normalization.md#function-settrainingmode)**(const bool isTrainging)<br>Sets the layer's training mode.  |

## Detailed Description

```cpp
class nn::BatchNormalization;
```

Implements Batch Normalization layer. 

Batch Normalization is a technique used to normalize the inputs of each layer to improve the training speed and stability of neural networks. 

## Public Functions Documentation

### function BatchNormalization

```cpp
BatchNormalization(
    const int numFeatures,
    const double momentum =0.99,
    const double epsilon =1e-15
)
```

Constructs a [BatchNormalization](classnn_1_1_batch_normalization.md) layer. 

**Parameters**: 

  * **numFeatures** Number of features (input dimensions). 
  * **momentum** Momentum for updating running mean and variance (default: 0.99). 
  * **epsilon** Small constant for numerical stability (default: 1e-15). 


### function BatchNormalization

```cpp
BatchNormalization(
    std::ifstream & file
)
```

Constructs a [BatchNormalization](classnn_1_1_batch_normalization.md) layer from a file. 

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


**Return**: The output matrix after applying [BatchNormalization](classnn_1_1_batch_normalization.md). 

**Reimplements**: [nn::Layer::forward](classnn_1_1_layer.md#function-forward)


### function backward

```cpp
virtual Matrix backward(
    const Matrix & gradient,
    Optimizer & optimizer
) override
```

Performs backward propagation. 

**Parameters**: 

  * **gradient** The gradient of the loss with respect to the output.
  * **optimizer** The optimizer to use for weights and biases updates.


**Return**: The gradient of the loss with respect to the input. 

**Reimplements**: [nn::Layer::backward](classnn_1_1_layer.md#function-backward)


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


### function setTrainingMode

```cpp
inline void setTrainingMode(
    const bool isTrainging
)
```

Sets the layer's training mode. 

**Parameters**: 

  * **isTraining** True for training, false for inference. 
