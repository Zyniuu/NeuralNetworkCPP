# nn::Optimizer



Abstract base class for optimizers. 


`#include <Optimizer.hpp>`

Inherited by [nn::Adam](classnn_1_1_adam.md), [nn::RMSprop](classnn_1_1_r_m_sprop.md), [nn::SGD](classnn_1_1_s_g_d.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[Optimizer](classnn_1_1_optimizer.md#function-optimizer)**(double learningRate)<br>Constructs an optimizer.  |
| virtual void | **[update](classnn_1_1_optimizer.md#function-update)**([Matrix](classnn_1_1_matrix.md) & weights, [Matrix](classnn_1_1_matrix.md) & biases, const [Matrix](classnn_1_1_matrix.md) & gradWeights, const [Matrix](classnn_1_1_matrix.md) & gradBiases) =0<br>Updates the weights and biases of a layer.  |

## Protected Attributes

|                | Name           |
| -------------- | -------------- |
| double | **[m_learningRate](classnn_1_1_optimizer.md#variable-m_learningrate)** <br>Learning rate for parameter updates.  |

## Public Functions Documentation

### function Optimizer

```cpp
inline Optimizer(
    double learningRate
)
```

Constructs an optimizer. 

**Parameters**: 

  * **learningRate** The learning rate. 


### function update

```cpp
virtual void update(
    Matrix & weights,
    Matrix & biases,
    const Matrix & gradWeights,
    const Matrix & gradBiases
) =0
```

Updates the weights and biases of a layer. 

**Parameters**: 

  * **weights** The weight matrix to update. 
  * **biases** The bias vector to update. 
  * **gradWeights** The gradient of the loss with respect to the weights. 
  * **gradBiases** The gradient of the loss with respect to the biases. 


**Reimplemented by**: [nn::Adam::update](classnn_1_1_adam.md#function-update), [nn::RMSprop::update](classnn_1_1_r_m_sprop.md#function-update), [nn::SGD::update](classnn_1_1_s_g_d.md#function-update)


## Protected Attributes Documentation

### variable m_learningRate

```cpp
double m_learningRate;
```

Learning rate for parameter updates. 
