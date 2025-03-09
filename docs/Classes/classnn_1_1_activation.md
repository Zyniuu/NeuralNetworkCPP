# nn::Activation



Abstract base class for activation functions. 


`#include <Activation.hpp>`

Inherited by [nn::ReLU](classnn_1_1_re_l_u.md), [nn::Sigmoid](classnn_1_1_sigmoid.md), [nn::Softmax](classnn_1_1_softmax.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| virtual [Matrix](classnn_1_1_matrix.md) | **[forward](classnn_1_1_activation.md#function-forward)**(const [Matrix](classnn_1_1_matrix.md) & input) =0<br>Applies the activation function to the input.  |
| virtual [Matrix](classnn_1_1_matrix.md) | **[backward](classnn_1_1_activation.md#function-backward)**(const [Matrix](classnn_1_1_matrix.md) & gradient) =0<br>Computes the gradient of the activation function.  |

## Protected Attributes

|                | Name           |
| -------------- | -------------- |
| [Matrix](classnn_1_1_matrix.md) | **[m_output](classnn_1_1_activation.md#variable-m_output)** <br>Stores the output of the forward pass for use in the backward pass.  |

## Public Functions Documentation

### function forward

```cpp
virtual Matrix forward(
    const Matrix & input
) =0
```

Applies the activation function to the input. 

**Parameters**: 

  * **input** The input matrix. 


**Return**: The output matrix. 

**Reimplemented by**: [nn::ReLU::forward](classnn_1_1_re_l_u.md#function-forward), [nn::Sigmoid::forward](classnn_1_1_sigmoid.md#function-forward), [nn::Softmax::forward](classnn_1_1_softmax.md#function-forward)


### function backward

```cpp
virtual Matrix backward(
    const Matrix & gradient
) =0
```

Computes the gradient of the activation function. 

**Parameters**: 

  * **gradient** The gradient of the loss with respect to the output. 


**Return**: The gradient of the loss with respect to the input. 

**Reimplemented by**: [nn::ReLU::backward](classnn_1_1_re_l_u.md#function-backward), [nn::Sigmoid::backward](classnn_1_1_sigmoid.md#function-backward), [nn::Softmax::backward](classnn_1_1_softmax.md#function-backward)


## Protected Attributes Documentation

### variable m_output

```cpp
Matrix m_output;
```

Stores the output of the forward pass for use in the backward pass. 
