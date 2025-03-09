# nn::BinaryCrossEntropy



Implements the Binary Cross-Entropy loss function.  [More...](#detailed-description)


`#include <BinaryCrossEntropy.hpp>`

Inherits from [nn::Loss](classnn_1_1_loss.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| virtual double | **[computeLoss](classnn_1_1_binary_cross_entropy.md#function-computeloss)**(const [Matrix](classnn_1_1_matrix.md) & predictions, const [Matrix](classnn_1_1_matrix.md) & targets) override<br>Computes the loss between predictions and targets using BCE formula.  |
| virtual [Matrix](classnn_1_1_matrix.md) | **[computeGradient](classnn_1_1_binary_cross_entropy.md#function-computegradient)**(const [Matrix](classnn_1_1_matrix.md) & predictions, const [Matrix](classnn_1_1_matrix.md) & targets) override<br>Computes the gradient of the loss with respect to predictions.  |

## Additional inherited members

**Protected Attributes inherited from [nn::Loss](classnn_1_1_loss.md)**

|                | Name           |
| -------------- | -------------- |
| double | **[m_epsilon](classnn_1_1_loss.md#variable-m_epsilon)** <br>Small number for numerical stability.  |


## Detailed Description

```cpp
class nn::BinaryCrossEntropy;
```

Implements the Binary Cross-Entropy loss function. 

Binary Cross-Entropy is used for binary classification tasks. It measures the difference between predicted probabilities and true binary labels. 

## Public Functions Documentation

### function computeLoss

```cpp
virtual double computeLoss(
    const Matrix & predictions,
    const Matrix & targets
) override
```

Computes the loss between predictions and targets using BCE formula. 

**Parameters**: 

  * **predictions** The predicted values. 
  * **targets** The target values. 


**Return**: The computed loss. 

**Reimplements**: [nn::Loss::computeLoss](classnn_1_1_loss.md#function-computeloss)


### function computeGradient

```cpp
virtual Matrix computeGradient(
    const Matrix & predictions,
    const Matrix & targets
) override
```

Computes the gradient of the loss with respect to predictions. 

**Parameters**: 

  * **predictions** The predicted values. 
  * **targets** The target values. 


**Return**: The gradient of the loss. 

**Reimplements**: [nn::Loss::computeGradient](classnn_1_1_loss.md#function-computegradient)
