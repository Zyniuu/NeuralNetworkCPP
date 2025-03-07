# nn::ModelEvaluator



Handles model evaluation, prediction, and metric computation.  [More...](#detailed-description)


`#include <ModelEvaluator.hpp>`

Inherits from [nn::ModelLayers](classnn_1_1_model_layers.md)

Inherited by [nn::ModelTrainer](classnn_1_1_model_trainer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| std::vector< double > | **[predict](classnn_1_1_model_evaluator.md#function-predict)**(const std::vector< double > & input)<br>Predicts the output for a given input.  |
| std::vector< std::vector< double > > | **[predict](classnn_1_1_model_evaluator.md#function-predict)**(const std::vector< std::vector< double > > & input)<br>Predicts the output for a given vector of inputs.  |
| double | **[evaluate](classnn_1_1_model_evaluator.md#function-evaluate)**(const std::vector< std::vector< double > > & xTest, const std::vector< std::vector< double > > & yTest, const [e_metric](../Namespaces/namespacenn.md#enum-e_metric) metric =[ACCURACY_LOG](../Namespaces/namespacenn.md#enum-e_metric))<br>Evaluates the model on the provided test data.  |

## Protected Functions

|                | Name           |
| -------------- | -------------- |
| [Matrix](classnn_1_1_matrix.md) | **[forward](classnn_1_1_model_evaluator.md#function-forward)**(const [Matrix](classnn_1_1_matrix.md) & input)<br>Performs forward propagation through the network.  |
| std::vector< double > | **[evaluate](classnn_1_1_model_evaluator.md#function-evaluate)**(const std::vector< std::vector< double > > & xTest, const std::vector< std::vector< double > > & yTest, const std::vector< [e_metric](../Namespaces/namespacenn.md#enum-e_metric) > & metrics)<br>Evaluates the model on the provided test data.  |

## Additional inherited members

**Public Functions inherited from [nn::ModelLayers](classnn_1_1_model_layers.md)**

|                | Name           |
| -------------- | -------------- |
| void | **[addLayer](classnn_1_1_model_layers.md#function-addlayer)**(std::unique_ptr< [Layer](classnn_1_1_layer.md) > layer)<br>Adds a layer to the network.  |

**Protected Functions inherited from [nn::ModelLayers](classnn_1_1_model_layers.md)**

|                | Name           |
| -------------- | -------------- |
| void | **[initLayer](classnn_1_1_model_layers.md#function-initlayer)**([e_layerType](../Namespaces/namespacenn.md#enum-e_layertype) layerType, std::ifstream & file)<br>Initializes a layer based on the provided layer type.  |

**Protected Attributes inherited from [nn::ModelLayers](classnn_1_1_model_layers.md)**

|                | Name           |
| -------------- | -------------- |
| std::vector< std::unique_ptr< [Layer](classnn_1_1_layer.md) > > | **[m_layers](classnn_1_1_model_layers.md#variable-m_layers)** <br>Vector of layers in the network.  |


## Detailed Description

```cpp
class nn::ModelEvaluator;
```

Handles model evaluation, prediction, and metric computation. 

This class extends [ModelLayers](classnn_1_1_model_layers.md) and provides functionality for making predictions, evaluating the model, and computing metrics like accuracy and mean absolute error. 

## Public Functions Documentation

### function predict

```cpp
std::vector< double > predict(
    const std::vector< double > & input
)
```

Predicts the output for a given input. 

**Parameters**: 

  * **input** The input vector. 


**Return**: The predicted output vector. 

### function predict

```cpp
std::vector< std::vector< double > > predict(
    const std::vector< std::vector< double > > & input
)
```

Predicts the output for a given vector of inputs. 

**Parameters**: 

  * **input** The vector of vector of inputs. 


**Return**: The predicted vector of vector of outputs. 

### function evaluate

```cpp
double evaluate(
    const std::vector< std::vector< double > > & xTest,
    const std::vector< std::vector< double > > & yTest,
    const e_metric metric =ACCURACY_LOG
)
```

Evaluates the model on the provided test data. 

**Parameters**: 

  * **xTest** Test data (vector of input vectors). 
  * **yTest** Test labels (vector of output vectors). 
  * **metric** The metric to compute (default: ACCURACY_LOG). 


**Return**: The computed metric. 

## Protected Functions Documentation

### function forward

```cpp
Matrix forward(
    const Matrix & input
)
```

Performs forward propagation through the network. 

**Parameters**: 

  * **input** The input matrix. 


**Return**: The output matrix. 

### function evaluate

```cpp
std::vector< double > evaluate(
    const std::vector< std::vector< double > > & xTest,
    const std::vector< std::vector< double > > & yTest,
    const std::vector< e_metric > & metrics
)
```

Evaluates the model on the provided test data. 

**Parameters**: 

  * **xTest** Test data (vector of input vectors). 
  * **yTest** Test labels (vector of output vectors). 
  * **metrics** The vector of metrics to compute. 


**Return**: The vector of computed metrics. 
