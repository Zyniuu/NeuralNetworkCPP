# nn::Adam



[Adam]() optimizer.  [More...](#detailed-description)


`#include <Adam.hpp>`

Inherits from [nn::Optimizer](classnn_1_1_optimizer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[Adam](classnn_1_1_adam.md#function-adam)**(double learningRate =0.001, double beta1 =0.9, double beta2 =0.999, double epsilon =1e-8)<br>Constructs an [Adam](classnn_1_1_adam.md) optimizer.  |
| virtual void | **[update](classnn_1_1_adam.md#function-update)**([Matrix](classnn_1_1_matrix.md) & weights, [Matrix](classnn_1_1_matrix.md) & biases, const [Matrix](classnn_1_1_matrix.md) & gradWeights, const [Matrix](classnn_1_1_matrix.md) & gradBiases) override<br>Updates the weights and biases using [Adam](classnn_1_1_adam.md).  |

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
class nn::Adam;
```

[Adam]() optimizer. 

This optimizer combines the benefits of momentum and [RMSprop](classnn_1_1_r_m_sprop.md) to achieve faster convergence and better performance on a wide range of problems. 

## Public Functions Documentation

### function Adam

```cpp
Adam(
    double learningRate =0.001,
    double beta1 =0.9,
    double beta2 =0.999,
    double epsilon =1e-8
)
```

Constructs an [Adam](classnn_1_1_adam.md) optimizer. 

**Parameters**: 

  * **learningRate** The learning rate (default: 0.001). 
  * **beta1** Exponential decay rate for the first moment estimates (default: 0.9). 
  * **beta2** Exponential decay rate for the second moment estimates (default: 0.999). 
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

Updates the weights and biases using [Adam](classnn_1_1_adam.md). 

**Parameters**: 

  * **weights** The weight matrix to update. 
  * **biases** The bias vector to update. 
  * **gradWeights** The gradient of the loss with respect to the weights. 
  * **gradBiases** The gradient of the loss with respect to the biases. 


**Reimplements**: [nn::Optimizer::update](classnn_1_1_optimizer.md#function-update)
