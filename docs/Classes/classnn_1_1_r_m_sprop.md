# nn::RMSprop



[RMSprop]() optimizer.  [More...](#detailed-description)


`#include <RMSprop.hpp>`

Inherits from [nn::Optimizer](classnn_1_1_optimizer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[RMSprop](classnn_1_1_r_m_sprop.md#function-rmsprop)**(double learningRate =0.001, double gamma =0.9, double epsilon =1e-8)<br>Constructs an [RMSprop](classnn_1_1_r_m_sprop.md) optimizer.  |
| virtual void | **[update](classnn_1_1_r_m_sprop.md#function-update)**([Matrix](classnn_1_1_matrix.md) & weights, [Matrix](classnn_1_1_matrix.md) & biases, const [Matrix](classnn_1_1_matrix.md) & gradWeights, const [Matrix](classnn_1_1_matrix.md) & gradBiases) override<br>Updates the weights and biases using [RMSprop](classnn_1_1_r_m_sprop.md).  |

## Additional inherited members

**Public Functions inherited from [nn::Optimizer](classnn_1_1_optimizer.md)**

|                | Name           |
| -------------- | -------------- |
| | **[Optimizer](classnn_1_1_optimizer.md#function-optimizer)**(double learningRate)<br>Constructs an optimizer.  |

**Protected Attributes inherited from [nn::Optimizer](classnn_1_1_optimizer.md)**

|                | Name           |
| -------------- | -------------- |
| double | **[m_learningRate](classnn_1_1_optimizer.md#variable-m_learningrate)** <br>Learning rate for parameter updates.  |


## Detailed Description

```cpp
class nn::RMSprop;
```

[RMSprop]() optimizer. 

This optimizer divides the learning rate by an exponentially decaying average of squared gradients to normalize the updates. 

## Public Functions Documentation

### function RMSprop

```cpp
RMSprop(
    double learningRate =0.001,
    double gamma =0.9,
    double epsilon =1e-8
)
```

Constructs an [RMSprop](classnn_1_1_r_m_sprop.md) optimizer. 

**Parameters**: 

  * **learningRate** The learning rate (default: 0.001). 
  * **gamma** Decay rate for the moving average (default: 0.9). 
  * **epsilon** Small constant for numerical stability (default: 1e-8). 


### function update

```cpp
virtual void update(
    Matrix & weights,
    Matrix & biases,
    const Matrix & gradWeights,
    const Matrix & gradBiases
) override
```

Updates the weights and biases using [RMSprop](classnn_1_1_r_m_sprop.md). 

**Parameters**: 

  * **weights** The weight matrix to update. 
  * **biases** The bias vector to update. 
  * **gradWeights** The gradient of the loss with respect to the weights. 
  * **gradBiases** The gradient of the loss with respect to the biases. 


**Reimplements**: [nn::Optimizer::update](classnn_1_1_optimizer.md#function-update)
