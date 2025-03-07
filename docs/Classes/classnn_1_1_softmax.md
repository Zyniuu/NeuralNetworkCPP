# nn::Softmax



Implements the [Softmax]() activation function.  [More...](#detailed-description)


`#include <Softmax.hpp>`

Inherits from [nn::Activation](classnn_1_1_activation.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| virtual [Matrix](classnn_1_1_matrix.md) | **[forward](classnn_1_1_softmax.md#function-forward)**(const [Matrix](classnn_1_1_matrix.md) & input) override<br>Applies the [Softmax](classnn_1_1_softmax.md) function to the input matrix.  |
| virtual [Matrix](classnn_1_1_matrix.md) | **[backward](classnn_1_1_softmax.md#function-backward)**(const [Matrix](classnn_1_1_matrix.md) & gradient) override<br>Computes the gradient of the [Softmax](classnn_1_1_softmax.md) function.  |

## Additional inherited members

**Protected Attributes inherited from [nn::Activation](classnn_1_1_activation.md)**

|                | Name           |
| -------------- | -------------- |
| [Matrix](classnn_1_1_matrix.md) | **[m_output](classnn_1_1_activation.md#variable-m_output)** <br>Stores the output of the forward pass for use in the backward pass.  |


## Detailed Description

```cpp
class nn::Softmax;
```

Implements the [Softmax]() activation function. 

[Softmax](classnn_1_1_softmax.md) is defined as: exp(x_i) / {sum_{j} exp(x_j)} 

## Public Functions Documentation

### function forward

```cpp
virtual Matrix forward(
    const Matrix & input
) override
```

Applies the [Softmax](classnn_1_1_softmax.md) function to the input matrix. 

**Parameters**: 

  * **input** The input matrix. 


**Return**: The output matrix after applying [Softmax](classnn_1_1_softmax.md). 

**Reimplements**: [nn::Activation::forward](classnn_1_1_activation.md#function-forward)


### function backward

```cpp
virtual Matrix backward(
    const Matrix & gradient
) override
```

Computes the gradient of the [Softmax](classnn_1_1_softmax.md) function. 

**Parameters**: 

  * **gradient** The gradient of the loss with respect to the output. 


**Return**: The gradient of the loss with respect to the input. 

**Reimplements**: [nn::Activation::backward](classnn_1_1_activation.md#function-backward)
