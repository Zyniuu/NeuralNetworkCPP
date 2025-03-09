# nn::ModelTrainer



Handles the training of a neural network model.  [More...](#detailed-description)


`#include <ModelTrainer.hpp>`

Inherits from [nn::ModelEvaluator](classnn_1_1_model_evaluator.md), [nn::ModelLayers](classnn_1_1_model_layers.md)

Inherited by [nn::NeuralNetworkCPP](classnn_1_1_neural_network_c_p_p.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| void | **[backward](classnn_1_1_model_trainer.md#function-backward)**(const [Matrix](classnn_1_1_matrix.md) & gradient)<br>Performs backward propagation through the network.  |
| void | **[compile](classnn_1_1_model_trainer.md#function-compile)**(std::unique_ptr< [Optimizer](classnn_1_1_optimizer.md) > optimizer, std::unique_ptr< [Loss](classnn_1_1_loss.md) > lossFunc, const std::vector< [e_metric](../Namespaces/namespacenn.md#enum-e_metric) > & metrics ={})<br>Compiles the model with the specified optimizer and loss function.  |
| bool | **[train](classnn_1_1_model_trainer.md#function-train)**(const std::vector< std::vector< double > > & xTrain, const std::vector< std::vector< double > > & yTrain, const int epochs, const int batchSize =1, const double validationSplit =0.0, const int patience =10, const double minDelta =0.0001, const bool verbose =true)<br>Trains the model on the provided data.  |

## Additional inherited members

**Public Functions inherited from [nn::ModelEvaluator](classnn_1_1_model_evaluator.md)**

|                | Name           |
| -------------- | -------------- |
| std::vector< double > | **[predict](classnn_1_1_model_evaluator.md#function-predict)**(const std::vector< double > & input)<br>Predicts the output for a given input.  |
| std::vector< std::vector< double > > | **[predict](classnn_1_1_model_evaluator.md#function-predict)**(const std::vector< std::vector< double > > & input)<br>Predicts the output for a given vector of inputs.  |
| double | **[evaluate](classnn_1_1_model_evaluator.md#function-evaluate)**(const std::vector< std::vector< double > > & xTest, const std::vector< std::vector< double > > & yTest, const [e_metric](../Namespaces/namespacenn.md#enum-e_metric) metric =[ACCURACY_LOG](../Namespaces/namespacenn.md#enum-e_metric))<br>Evaluates the model on the provided test data.  |

**Protected Functions inherited from [nn::ModelEvaluator](classnn_1_1_model_evaluator.md)**

|                | Name           |
| -------------- | -------------- |
| [Matrix](classnn_1_1_matrix.md) | **[forward](classnn_1_1_model_evaluator.md#function-forward)**(const [Matrix](classnn_1_1_matrix.md) & input)<br>Performs forward propagation through the network.  |
| std::vector< double > | **[evaluate](classnn_1_1_model_evaluator.md#function-evaluate)**(const std::vector< std::vector< double > > & xTest, const std::vector< std::vector< double > > & yTest, const std::vector< [e_metric](../Namespaces/namespacenn.md#enum-e_metric) > & metrics)<br>Evaluates the model on the provided test data.  |

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
class nn::ModelTrainer;
```

Handles the training of a neural network model. 

This class extends [ModelEvaluator](classnn_1_1_model_evaluator.md) and provides functionality for compiling and training the model using an optimizer, loss function, and metrics. 

## Public Functions Documentation

### function backward

```cpp
void backward(
    const Matrix & gradient
)
```

Performs backward propagation through the network. 

**Parameters**: 

  * **gradient** The gradient of the loss with respect to the output. 


### function compile

```cpp
void compile(
    std::unique_ptr< Optimizer > optimizer,
    std::unique_ptr< Loss > lossFunc,
    const std::vector< e_metric > & metrics ={}
)
```

Compiles the model with the specified optimizer and loss function. 

**Parameters**: 

  * **optimizer** The optimizer to use for training. 
  * **lossFunc** The loss function to use for training. 
  * **metrics** The vector of metrics to display (default: {}). 


### function train

```cpp
bool train(
    const std::vector< std::vector< double > > & xTrain,
    const std::vector< std::vector< double > > & yTrain,
    const int epochs,
    const int batchSize =1,
    const double validationSplit =0.0,
    const int patience =10,
    const double minDelta =0.0001,
    const bool verbose =true
)
```

Trains the model on the provided data. 

**Parameters**: 

  * **xTrain** Training data (vector of input vectors). 
  * **yTrain** Training labels (vector of output vectors). 
  * **epochs** Number of training epochs. 
  * **batchSize** Size of each training batch (default: 1). 
  * **validationSplit** Fraction of the data to use for validation (default: 0.0). 
  * **patience** Number of epochs to wait for improvement (default: 10). 
  * **minDelta** Minimum improvement to reset patience (default: 0.0001). 
  * **verbose** If true, logs will be displayed (default: true). 


**Return**: True if the training has been completed, false if stopped early 
