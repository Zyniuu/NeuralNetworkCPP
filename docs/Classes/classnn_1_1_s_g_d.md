# nn::SGD



Stochastic Gradient Descent ([SGD]()) optimizer with momentum.  [More...](#detailed-description)


`#include <SGD.hpp>`

Inherits from [nn::Optimizer](classnn_1_1_optimizer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[SGD](classnn_1_1_s_g_d.md#function-sgd)**(double learningRate =0.001, double momentum =0.9)<br>Constructs an [SGD](classnn_1_1_s_g_d.md) optimizer.  |
| virtual void | **[update](classnn_1_1_s_g_d.md#function-update)**([Matrix](classnn_1_1_matrix.md) & weights, [Matrix](classnn_1_1_matrix.md) & biases, const [Matrix](classnn_1_1_matrix.md) & gradWeights, const [Matrix](classnn_1_1_matrix.md) & gradBiases) override<br>Updates the weights and biases using momentum.  |

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
class nn::SGD;
```

Stochastic Gradient Descent ([SGD]()) optimizer with momentum. 

This optimizer updates parameters using the gradient of the loss function and a momentum term to accelerate convergence. 

## Public Functions Documentation

### function SGD

```cpp
SGD(
    double learningRate =0.001,
    double momentum =0.9
)
```

Constructs an [SGD](classnn_1_1_s_g_d.md) optimizer. 

**Parameters**: 

  * **learningRate** The learning rate (default: 0.001). 
  * **momentum** The momentum factor (default: 0.9). 


### function update

```cpp
virtual void update(
    Matrix & weights,
    Matrix & biases,
    const Matrix & gradWeights,
    const Matrix & gradBiases
) override
```

Updates the weights and biases using momentum. 

**Parameters**: 

  * **weights** The weight matrix to update. 
  * **biases** The bias vector to update. 
  * **gradWeights** The gradient of the loss with respect to the weights. 
  * **gradBiases** The gradient of the loss with respect to the biases. 


**Reimplements**: [nn::Optimizer::update](classnn_1_1_optimizer.md#function-update)
