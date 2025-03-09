# nn::NeuralNetworkCPP



A neural network model that supports adding layers, compiling, training, and saving/loading.  [More...](#detailed-description)


`#include <NeuralNetworkCPP.hpp>`

Inherits from [nn::ModelTrainer](classnn_1_1_model_trainer.md), [nn::ModelEvaluator](classnn_1_1_model_evaluator.md), [nn::ModelLayers](classnn_1_1_model_layers.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[NeuralNetworkCPP](classnn_1_1_neural_network_c_p_p.md#function-neuralnetworkcpp)**(const int numThreads =std::thread::hardware_concurrency())<br>Default constructor.  |
| | **[NeuralNetworkCPP](classnn_1_1_neural_network_c_p_p.md#function-neuralnetworkcpp)**(const std::string & filename, const int numThreads =std::thread::hardware_concurrency())<br>Constructs a model from a saved file.  |
| void | **[save](classnn_1_1_neural_network_c_p_p.md#function-save)**(const std::string & filename) const<br>Saves the model to a file.  |

## Additional inherited members

**Public Functions inherited from [nn::ModelTrainer](classnn_1_1_model_trainer.md)**

|                | Name           |
| -------------- | -------------- |
| void | **[backward](classnn_1_1_model_trainer.md#function-backward)**(const [Matrix](classnn_1_1_matrix.md) & gradient)<br>Performs backward propagation through the network.  |
| void | **[compile](classnn_1_1_model_trainer.md#function-compile)**(std::unique_ptr< [Optimizer](classnn_1_1_optimizer.md) > optimizer, std::unique_ptr< [Loss](classnn_1_1_loss.md) > lossFunc, const std::vector< [e_metric](../Namespaces/namespacenn.md#enum-e_metric) > & metrics ={})<br>Compiles the model with the specified optimizer and loss function.  |
| bool | **[train](classnn_1_1_model_trainer.md#function-train)**(const std::vector< std::vector< double > > & xTrain, const std::vector< std::vector< double > > & yTrain, const int epochs, const int batchSize =1, const double validationSplit =0.0, const int patience =10, const double minDelta =0.0001, const bool verbose =true)<br>Trains the model on the provided data.  |

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
| std::vector< std::unique_ptr< [Layer](classnn_1_1_layer.md) > > | **[m_layers](classnn_1_1_model_layers.md#variable_m-layers)** <br>Vector of layers in the network.  |


## Detailed Description

```cpp
class nn::NeuralNetworkCPP;
```

A neural network model that supports adding layers, compiling, training, and saving/loading. 

This class provides a high-level interface for building and training neural networks. 

## Public Functions Documentation

### function NeuralNetworkCPP

```cpp
NeuralNetworkCPP(
    const int numThreads =std::thread::hardware_concurrency()
)
```

Default constructor. 

**Parameters**: 

  * **numThreads** Number of threads in the thread pool (default: hardware concurrency). 


### function NeuralNetworkCPP

```cpp
NeuralNetworkCPP(
    const std::string & filename,
    const int numThreads =std::thread::hardware_concurrency()
)
```

Constructs a model from a saved file. 

**Parameters**: 

  * **filename** Path to the file containing the saved model. 
  * **numThreads** Number of threads in the thread pool (default: hardware concurrency). 


**Exceptions**: 

  * **std::runtime_error** If the file cannot be opened or is invalid. 


### function save

```cpp
void save(
    const std::string & filename
) const
```

Saves the model to a file. 

**Parameters**: 

  * **filename** Path to the file where the model will be saved. 


**Exceptions**: 

  * **std::runtime_error** If the file cannot be opened or writing fails. 
