# nn::Loss



Abstract base class for loss functions.  [More...](#detailed-description)


`#include <Loss.hpp>`

Inherited by [nn::BinaryCrossEntropy](classnn_1_1_binary_cross_entropy.md), [nn::CategoricalCrossEntropy](classnn_1_1_categorical_cross_entropy.md), [nn::MeanSquaredError](classnn_1_1_mean_squared_error.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| virtual double | **[computeLoss](classnn_1_1_loss.md#function-computeloss)**(const [Matrix](classnn_1_1_matrix.md) & predictions, const [Matrix](classnn_1_1_matrix.md) & targets) =0<br>Computes the loss between predictions and targets.  |
| virtual [Matrix](classnn_1_1_matrix.md) | **[computeGradient](classnn_1_1_loss.md#function-computegradient)**(const [Matrix](classnn_1_1_matrix.md) & predictions, const [Matrix](classnn_1_1_matrix.md) & targets) =0<br>Computes the gradient of the loss with respect to predictions.  |

## Protected Attributes

|                | Name           |
| -------------- | -------------- |
| double | **[m_epsilon](classnn_1_1_loss.md#variable-m_epsilon)** <br>Small number for numerical stability.  |

## Detailed Description

```cpp
class nn::Loss;
```

Abstract base class for loss functions. 

This class defines the interface for computing loss and its gradient. 

## Public Functions Documentation

### function computeLoss

```cpp
virtual double computeLoss(
    const Matrix & predictions,
    const Matrix & targets
) =0
```

Computes the loss between predictions and targets. 

**Parameters**: 

  * **predictions** The predicted values. 
  * **targets** The target values. 


**Return**: The computed loss. 

**Reimplemented by**: [nn::BinaryCrossEntropy::computeLoss](classnn_1_1_binary_cross_entropy.md#function-computeloss), [nn::CategoricalCrossEntropy::computeLoss](classnn_1_1_categorical_cross_entropy.md#function-computeloss), [nn::MeanSquaredError::computeLoss](classnn_1_1_mean_squared_error.md#function-computeloss)


### function computeGradient

```cpp
virtual Matrix computeGradient(
    const Matrix & predictions,
    const Matrix & targets
) =0
```

Computes the gradient of the loss with respect to predictions. 

**Parameters**: 

  * **predictions** The predicted values. 
  * **targets** The target values. 


**Return**: The gradient of the loss. 

**Reimplemented by**: [nn::BinaryCrossEntropy::computeGradient](classnn_1_1_binary_cross_entropy.md#function-computegradient), [nn::CategoricalCrossEntropy::computeGradient](classnn_1_1_categorical_cross_entropy.md#function-computegradient), [nn::MeanSquaredError::computeGradient](classnn_1_1_mean_squared_error.md#function-computegradient)


## Protected Attributes Documentation

### variable m_epsilon

```cpp
double m_epsilon = 1e-15;
```

Small number for numerical stability. 
