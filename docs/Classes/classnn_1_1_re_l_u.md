# nn::ReLU



Implements the Rectified Linear Unit ([ReLU]()) activation function.  [More...](#detailed-description)


`#include <ReLU.hpp>`

Inherits from [nn::Activation](classnn_1_1_activation.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| virtual [Matrix](classnn_1_1_matrix.md) | **[forward](classnn_1_1_re_l_u.md#function-forward)**(const [Matrix](classnn_1_1_matrix.md) & input) override<br>Applies the [ReLU](classnn_1_1_re_l_u.md) function to the input matrix.  |
| virtual [Matrix](classnn_1_1_matrix.md) | **[backward](classnn_1_1_re_l_u.md#function-backward)**(const [Matrix](classnn_1_1_matrix.md) & gradient) override<br>Computes the gradient of the [ReLU](classnn_1_1_re_l_u.md) function.  |

## Additional inherited members

**Protected Attributes inherited from [nn::Activation](classnn_1_1_activation.md)**

|                | Name           |
| -------------- | -------------- |
| [Matrix](classnn_1_1_matrix.md) | **[m_output](classnn_1_1_activation.md#variable-m_output)** <br>Stores the output of the forward pass for use in the backward pass.  |


## Detailed Description

```cpp
class nn::ReLU;
```

Implements the Rectified Linear Unit ([ReLU]()) activation function. 

[ReLU](classnn_1_1_re_l_u.md) is defined as: ReLU(x) = max(0, x) 

## Public Functions Documentation

### function forward

```cpp
virtual Matrix forward(
    const Matrix & input
) override
```

Applies the [ReLU](classnn_1_1_re_l_u.md) function to the input matrix. 

**Parameters**: 

  * **input** The input matrix. 


**Return**: The output matrix after applying [ReLU](classnn_1_1_re_l_u.md). 

**Reimplements**: [nn::Activation::forward](classnn_1_1_activation.md#function-forward)


### function backward

```cpp
virtual Matrix backward(
    const Matrix & gradient
) override
```

Computes the gradient of the [ReLU](classnn_1_1_re_l_u.md) function. 

**Parameters**: 

  * **gradient** The gradient of the loss with respect to the output. 


**Return**: The gradient of the loss with respect to the input. 

**Reimplements**: [nn::Activation::backward](classnn_1_1_activation.md#function-backward)
