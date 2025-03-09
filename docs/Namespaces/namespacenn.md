# nn

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::Activation](../Classes/classnn_1_1_activation.md)** <br>Abstract base class for activation functions.  |
| class | **[nn::Adam](../Classes/classnn_1_1_adam.md)** <br>[Adam]() optimizer.  |
| class | **[nn::BatchNormalization](../Classes/classnn_1_1_batch_normalization.md)** <br>Implements Batch Normalization layer.  |
| class | **[nn::BinaryCrossEntropy](../Classes/classnn_1_1_binary_cross_entropy.md)** <br>Implements the Binary Cross-Entropy loss function.  |
| class | **[nn::CategoricalCrossEntropy](../Classes/classnn_1_1_categorical_cross_entropy.md)** <br>Implements the Categorical Cross-Entropy loss function.  |
| class | **[nn::ColWiseProxy](../Classes/classnn_1_1_col_wise_proxy.md)** <br>Proxy class for performing column-wise operations on a matrix.  |
| class | **[nn::CSVReader](../Classes/classnn_1_1_c_s_v_reader.md)** <br>Reads data from a CSV file and separates it into features and labels.  |
| class | **[nn::DenseLayer](../Classes/classnn_1_1_dense_layer.md)** <br>Implements a fully connected (dense) layer.  |
| class | **[nn::HeNormal](../Classes/classnn_1_1_he_normal.md)** <br>Implements He Normal initialization for neural network weights.  |
| class | **[nn::HeUniform](../Classes/classnn_1_1_he_uniform.md)** <br>Implements He Uniform initialization for neural network weights.  |
| class | **[nn::Initializer](../Classes/classnn_1_1_initializer.md)** <br>Abstract base class for weight initializers in neural networks.  |
| class | **[nn::Layer](../Classes/classnn_1_1_layer.md)** <br>Abstract base class for neural network layers.  |
| class | **[nn::Logger](../Classes/classnn_1_1_logger.md)** <br>Handles logging of training progress and metrics.  |
| class | **[nn::Loss](../Classes/classnn_1_1_loss.md)** <br>Abstract base class for loss functions.  |
| class | **[nn::Matrix](../Classes/classnn_1_1_matrix.md)** <br>Represents a mathematical matrix with element-wise operations.  |
| class | **[nn::MeanSquaredError](../Classes/classnn_1_1_mean_squared_error.md)** <br>Implements the Mean Squared Error (MSE) loss function.  |
| class | **[nn::MinMaxScaler](../Classes/classnn_1_1_min_max_scaler.md)** <br>Normalizes data to a specified range (default: [0, 1]).  |
| class | **[nn::ModelEvaluator](../Classes/classnn_1_1_model_evaluator.md)** <br>Handles model evaluation, prediction, and metric computation.  |
| class | **[nn::ModelLayers](../Classes/classnn_1_1_model_layers.md)** <br>Manages the layers of a neural network model.  |
| class | **[nn::ModelTrainer](../Classes/classnn_1_1_model_trainer.md)** <br>Handles the training of a neural network model.  |
| class | **[nn::NeuralNetworkCPP](../Classes/classnn_1_1_neural_network_c_p_p.md)** <br>A neural network model that supports adding layers, compiling, training, and saving/loading.  |
| class | **[nn::Optimizer](../Classes/classnn_1_1_optimizer.md)** <br>Abstract base class for optimizers.  |
| class | **[nn::ReLU](../Classes/classnn_1_1_re_l_u.md)** <br>Implements the Rectified Linear Unit ([ReLU]()) activation function.  |
| class | **[nn::RMSprop](../Classes/classnn_1_1_r_m_sprop.md)** <br>[RMSprop]() optimizer.  |
| class | **[nn::RowWiseProxy](../Classes/classnn_1_1_row_wise_proxy.md)** <br>Proxy class for performing row-wise operations on a matrix.  |
| class | **[nn::Scaler](../Classes/classnn_1_1_scaler.md)** <br>Abstract base class for all scalers.  |
| class | **[nn::SGD](../Classes/classnn_1_1_s_g_d.md)** <br>Stochastic Gradient Descent ([SGD]()) optimizer with momentum.  |
| class | **[nn::Sigmoid](../Classes/classnn_1_1_sigmoid.md)** <br>Implements the [Sigmoid]() activation function.  |
| class | **[nn::Softmax](../Classes/classnn_1_1_softmax.md)** <br>Implements the [Softmax]() activation function.  |
| class | **[nn::StandardScaler](../Classes/classnn_1_1_standard_scaler.md)** <br>Normalizes data to have a mean of 0 and a standard deviation of 1.  |
| class | **[nn::ThreadPool](../Classes/classnn_1_1_thread_pool.md)** <br>A thread pool implementation for executing tasks in parallel.  |
| class | **[nn::XavierNormal](../Classes/classnn_1_1_xavier_normal.md)** <br>Implements Xavier (Glorot) normal initialization for neural network weights.  |
| class | **[nn::XavierUniform](../Classes/classnn_1_1_xavier_uniform.md)** <br>Implements Xavier (Glorot) uniform initialization for neural network weights.  |

## Types

|                | Name           |
| -------------- | -------------- |
| enum| **[e_layerType](#enum-e_layertype)** { DENSE, BATCH_NORM}<br>Enum with available layer types.  |
| enum| **[e_initializer](#enum-e_initializer)** { HE_NORMAL, HE_UNIFORM, XAVIER_NORMAL, XAVIER_UNIFORM}<br>Enum with avaible initializers.  |
| enum| **[e_activation](#enum-e_activation)** { RELU, SIGMOID, SOFTMAX, NONE}<br>Enum with avaible activation functions.  |
| enum| **[e_metric](#enum-e_metric)** { ACCURACY_LOG, MAE_LOG}<br>Enum with avaible metrics.  |

## Functions

|                | Name           |
| -------------- | -------------- |
| void | **[initGlobalThreadPool](#function-initglobalthreadpool)**(int numThreads =std::thread::hardware_concurrency())<br>Initializes the global thread pool with the specified number of threads.  |
| [ThreadPool](../Classes/classnn_1_1_thread_pool.md) & | **[getGlobalThreadPool](#function-getglobalthreadpool)**()<br>Returns the global thread pool instance.  |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator*](#function-operator*)**(const [ColWiseProxy](../Classes/classnn_1_1_col_wise_proxy.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator/](#function-operator/)**(const [ColWiseProxy](../Classes/classnn_1_1_col_wise_proxy.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator+](#function-operator+)**(const [ColWiseProxy](../Classes/classnn_1_1_col_wise_proxy.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator-](#function-operator-)**(const [ColWiseProxy](../Classes/classnn_1_1_col_wise_proxy.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| std::ostream & | **[operator<<](#function-operator<<)**(std::ostream & out, const [Matrix](../Classes/classnn_1_1_matrix.md) & m) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator*](#function-operator*)**(const [Matrix](../Classes/classnn_1_1_matrix.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator*](#function-operator*)**(const double scalar, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator*](#function-operator*)**(const [Matrix](../Classes/classnn_1_1_matrix.md) & left, const double scalar) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator+](#function-operator+)**(const [Matrix](../Classes/classnn_1_1_matrix.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator+](#function-operator+)**(const double scalar, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator+](#function-operator+)**(const [Matrix](../Classes/classnn_1_1_matrix.md) & left, const double scalar) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator-](#function-operator-)**(const [Matrix](../Classes/classnn_1_1_matrix.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator-](#function-operator-)**(const [Matrix](../Classes/classnn_1_1_matrix.md) & right, const double scalar) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator-](#function-operator-)**(const double scalar, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator/](#function-operator/)**(const [Matrix](../Classes/classnn_1_1_matrix.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator/](#function-operator/)**(const [Matrix](../Classes/classnn_1_1_matrix.md) & right, const double scalar) |
| bool | **[operator==](#function-operator==)**(const [Matrix](../Classes/classnn_1_1_matrix.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| bool | **[operator!=](#function-operator!=)**(const [Matrix](../Classes/classnn_1_1_matrix.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator-](#function-operator-)**(const [RowWiseProxy](../Classes/classnn_1_1_row_wise_proxy.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| [Matrix](../Classes/classnn_1_1_matrix.md) | **[operator/](#function-operator/)**(const [RowWiseProxy](../Classes/classnn_1_1_row_wise_proxy.md) & left, const [Matrix](../Classes/classnn_1_1_matrix.md) & right) |
| std::vector< std::vector< double > > | **[slice](#function-slice)**(const std::vector< std::vector< double > > & data, const int start, const int end)<br>Slices a 2D vector from index `start` to `end` (exclusive).  |
| void | **[reorderRows](#function-reorderrows)**(std::vector< std::vector< double > > & data, const std::vector< int > & order)<br>Reorders the rows of a 2D vector based on the provided order.  |
| void | **[shuffleDataset](#function-shuffledataset)**(std::vector< std::vector< double > > & data, std::vector< std::vector< double > > & labels)<br>Shuffles the rows of the provided data and labels.  |
| std::vector< std::vector< double > > | **[to_categorical](#function-to_categorical)**(const std::vector< std::vector< double > > & data, int numClasses =0)<br>Converts a vector of class labels into one-hot encoded vectors.  |

## Attributes

|                | Name           |
| -------------- | -------------- |
| std::unique_ptr< [ThreadPool](../Classes/classnn_1_1_thread_pool.md) > | **[globalThreadPool](#variable-globalthreadpool)** <br>Global thread pool instance accessible throughout the program.  |

## Types Documentation

### enum e_layerType

| Enumerator | Value | Description |
| ---------- | ----- | ----------- |
| DENSE | |   |
| BATCH_NORM | |   |



Enum with available layer types. 

### enum e_initializer

| Enumerator | Value | Description |
| ---------- | ----- | ----------- |
| HE_NORMAL | |   |
| HE_UNIFORM | |   |
| XAVIER_NORMAL | |   |
| XAVIER_UNIFORM | |   |



Enum with avaible initializers. 

### enum e_activation

| Enumerator | Value | Description |
| ---------- | ----- | ----------- |
| RELU | |   |
| SIGMOID | |   |
| SOFTMAX | |   |
| NONE | |   |



Enum with avaible activation functions. 

### enum e_metric

| Enumerator | Value | Description |
| ---------- | ----- | ----------- |
| ACCURACY_LOG | |   |
| MAE_LOG | |   |



Enum with avaible metrics. 


## Functions Documentation

### function initGlobalThreadPool

```cpp
void initGlobalThreadPool(
    int numThreads =std::thread::hardware_concurrency()
)
```

Initializes the global thread pool with the specified number of threads. 

**Parameters**: 

  * **numThreads** The number of threads to create. Defaults to the number of hardware threads. 


### function getGlobalThreadPool

```cpp
ThreadPool & getGlobalThreadPool()
```

Returns the global thread pool instance. 

**Exceptions**: 

  * **std::runtime_error** If the thread pool is not initialized. 


**Return**: A reference to the global thread pool. 

### function operator*

```cpp
Matrix operator*(
    const ColWiseProxy & left,
    const Matrix & right
)
```


**Parameters**: 

  * **other** A column vector ([Matrix](../Classes/classnn_1_1_matrix.md) with 1 column). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a column vector or its row count does not match the original matrix. 


**Return**: A new [Matrix](../Classes/classnn_1_1_matrix.md) after the column-wise multiplication. 

### function operator/

```cpp
Matrix operator/(
    const ColWiseProxy & left,
    const Matrix & right
)
```


**Parameters**: 

  * **other** A column vector ([Matrix](../Classes/classnn_1_1_matrix.md) with 1 column). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a column vector or its row count does not match the original matrix. 


**Return**: A new [Matrix](../Classes/classnn_1_1_matrix.md) after the column-wise division. 

### function operator+

```cpp
Matrix operator+(
    const ColWiseProxy & left,
    const Matrix & right
)
```


**Parameters**: 

  * **other** A column vector ([Matrix](../Classes/classnn_1_1_matrix.md) with 1 column). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a column vector or its row count does not match the original matrix. 


**Return**: A new [Matrix](../Classes/classnn_1_1_matrix.md) after the column-wise addition. 

### function operator-

```cpp
Matrix operator-(
    const ColWiseProxy & left,
    const Matrix & right
)
```


**Parameters**: 

  * **other** A column vector ([Matrix](../Classes/classnn_1_1_matrix.md) with 1 column). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a column vector or its row count does not match the original matrix. 


**Return**: A new [Matrix](../Classes/classnn_1_1_matrix.md) after the column-wise subtraction. 

### function operator<<

```cpp
std::ostream & operator<<(
    std::ostream & out,
    const Matrix & m
)
```


### function operator*

```cpp
Matrix operator*(
    const Matrix & left,
    const Matrix & right
)
```


### function operator*

```cpp
Matrix operator*(
    const double scalar,
    const Matrix & right
)
```


### function operator*

```cpp
Matrix operator*(
    const Matrix & left,
    const double scalar
)
```


### function operator+

```cpp
Matrix operator+(
    const Matrix & left,
    const Matrix & right
)
```


### function operator+

```cpp
Matrix operator+(
    const double scalar,
    const Matrix & right
)
```


### function operator+

```cpp
Matrix operator+(
    const Matrix & left,
    const double scalar
)
```


### function operator-

```cpp
Matrix operator-(
    const Matrix & left,
    const Matrix & right
)
```


### function operator-

```cpp
Matrix operator-(
    const Matrix & right,
    const double scalar
)
```


### function operator-

```cpp
Matrix operator-(
    const double scalar,
    const Matrix & right
)
```


### function operator/

```cpp
Matrix operator/(
    const Matrix & left,
    const Matrix & right
)
```


### function operator/

```cpp
Matrix operator/(
    const Matrix & right,
    const double scalar
)
```


### function operator==

```cpp
bool operator==(
    const Matrix & left,
    const Matrix & right
)
```


### function operator!=

```cpp
bool operator!=(
    const Matrix & left,
    const Matrix & right
)
```


### function operator-

```cpp
Matrix operator-(
    const RowWiseProxy & left,
    const Matrix & right
)
```


**Parameters**: 

  * **other** A row vector ([Matrix](../Classes/classnn_1_1_matrix.md) with 1 row). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a row vector or its column count does not match the original matrix. 


**Return**: A new [Matrix](../Classes/classnn_1_1_matrix.md) after the row-wise subtraction. 

### function operator/

```cpp
Matrix operator/(
    const RowWiseProxy & left,
    const Matrix & right
)
```


**Parameters**: 

  * **other** A row vector ([Matrix](../Classes/classnn_1_1_matrix.md) with 1 row). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a row vector or its column count does not match the original matrix. 


**Return**: A new [Matrix](../Classes/classnn_1_1_matrix.md) after the row-wise division. 

### function slice

```cpp
std::vector< std::vector< double > > slice(
    const std::vector< std::vector< double > > & data,
    const int start,
    const int end
)
```

Slices a 2D vector from index `start` to `end` (exclusive). 

**Parameters**: 

  * **data** The input 2D vector to slice. 
  * **start** The starting index (inclusive). 
  * **end** The ending index (exclusive). 


**Exceptions**: 

  * **std::out_of_range** If `start` or `end` are out of bounds. 


**Return**: A new 2D vector containing the sliced data. 

### function reorderRows

```cpp
void reorderRows(
    std::vector< std::vector< double > > & data,
    const std::vector< int > & order
)
```

Reorders the rows of a 2D vector based on the provided order. 

**Parameters**: 

  * **data** The input 2D vector to reorder. 
  * **order** A vector of indices specifying the new order of rows. 


**Exceptions**: 

  * **std::out_of_range** If any index in `order` is out of bounds. 


### function shuffleDataset

```cpp
void shuffleDataset(
    std::vector< std::vector< double > > & data,
    std::vector< std::vector< double > > & labels
)
```

Shuffles the rows of the provided data and labels. 

**Parameters**: 

  * **data** A 2D data vector to shuffle. 
  * **labels** A 2D labels vector to shuffle. 


**Exceptions**: 

  * **std::runtime_error** If data and labels have different amount of rows. 


### function to_categorical

```cpp
std::vector< std::vector< double > > to_categorical(
    const std::vector< std::vector< double > > & data,
    int numClasses =0
)
```

Converts a vector of class labels into one-hot encoded vectors. 

**Parameters**: 

  * **data** The input data as a vector of vectors of doubles (class labels). 
  * **numClasses** The number of classes. If 0, it is determined automatically. 


**Return**: std::vector<std::vector<double>> The one-hot encoded data. 


## Attributes Documentation

### variable globalThreadPool

```cpp
std::unique_ptr< ThreadPool > globalThreadPool = nullptr;
```

Global thread pool instance accessible throughout the program. 

This is a singleton-like thread pool to avoid redundant thread creation and ensure efficient resource usage across the application. 
