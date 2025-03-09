# NeuralNetworkCPP
A lightweight, modular, and easy-to-use C++ neural network library designed for building, training, and evaluating neural networks. The library implements features such as: batch processing, early stopping and multi threading.

# Table of contents

* [Usage](#usage)
* [Data preprocessing](#data-preprocessing)
    * [CSVReader](#csvreader)
    * [MinMaxScaler](#minmaxscaler)
    * [StandardScaler](#standardscaler)
    * [to_categorical](#to_categorical)
* [Initializers](#initializers)
    * [Xavier/Glorot initializer](#xavierglorot-initializer)
    * [He initializer](#he-initializer)
* [Activation functions](#activation-functions)
    * [ReLU](#relu)
    * [Sigmoid](#sigmoid)
    * [Softmax](#softmax)
    * [None](#none)
* [Loss functions](#loss-functions)
    * [BinaryCrossEntropy](#binarycrossentropy)
    * [CategoricalCrossEntropy](#categoricalcrossentropy)
    * [MeanSquaredError](#meansquarederror)
* [Layers](#layers)
    * [DenseLayer](#denselayer)
    * [BatchNormalization](#batchnormalization)
* [Optimizers](#optimizers)
    * [SGD](#sgd)
    * [RMSprop](#rmsprop)
    * [Adam](#adam)

## Usage

Start by including the library to your project

```cpp
#include <NeuralNetworkCPP/NeuralNetworkCPP.hpp>
```

Creating a new model:

```cpp
nn::NeuralNetworkCPP model;
```

Or if you already have a file with the trained model:

```cpp
nn::NeuralNetworkCPP model("trained_model.bin");
```

Adding layers to the model:

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(2, 8, nn::HE_NORMAL, nn::RELU));
```

In above example we have added a Dense (fully connected) layer to our model with 2 input neurons, 8 output neurons, He normal initializer and ReLU activation function.

Next, before training we need to compile our model by providing optimizer, loss function and optionaly metrics to display during training:

```cpp
model.compile(
    std::make_unique<nn::Adam>(),
    std::make_unique<nn::BinaryCrossEntropy>(),
    { nn::ACCURACY_LOG }
);
```

Finally we can run the training by providing:
* training data and labels
* number of epochs
* batch size
* validation split
* patience count (how many epochs to wait for network to improve)
* minimum error delta
* whether the training should be verbose or not

```cpp
model.train(trainData, trainLabels, 10, 512, 0.2, 1, 0.00001, true);
```

Once your model has finished training you can evaluate it by providing:
* test data and labels
* metric to calculate (by default it calculates accuracy)

```cpp
model.evaluate(testData, testLabels, nn::MAE_LOG)
```

To use the model for prediction you can use either a vector of inputs or a vector of input vectors (in this case you will get a vector of output vectors):

```cpp
std::vector<std::vector<double>> data = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

// Using a vector of inputs
std::vector<double> output1 = model.predict({0.0, 0.0});

// Using a vector of input vectors
std::vector<std::vector<double>> model.predict(data);
```

You can save your trained model to the file using:

```cpp
model.save("trained_model.bin");
```

## Data preprocessing

A csv file reader ([CSVReader](docs/Classes/classnn_1_1_c_s_v_reader.md)) has been implemented into the project, as well as two scalers: [MinMaxScaler](docs/Classes/classnn_1_1_min_max_scaler.md) and [StandardScaler](docs/Classes/classnn_1_1_standard_scaler.md) for data normalization, and [to_categorical](docs/Namespaces/namespacenn.md#function-to_categorical) function for converting labels into one-hot encoded vectors.

### [CSVReader](docs/Classes/classnn_1_1_c_s_v_reader.md)

Start by including csv reader into your project:

```cpp
#include <NeuralNetworkCPP/DataPreprocessing/CSVReader/CSVReader.hpp>
```

Create a new reader by providing:
* filename
* separator
* if targets are at the end of each line
* if the file has a header to skip

```cpp
nn::CSVReader mnistTrain("mnist_train.csv", ',', false, true);
```

Next read the csv file contents:

```cpp
mnistTrain.read();
```

Now you can retrieve the features from the file, as well as labels:

```cpp
// Retrieve features
std::vector<std::vector<double>> data = mnistTrain.getData();

// Retrieve labels
std::vector<std::vector<double>> labels = mnistTrain.getLabels();
```

### [MinMaxScaler](docs/Classes/classnn_1_1_min_max_scaler.md)

Start by including [MinMaxScaler](docs/Classes/classnn_1_1_min_max_scaler.md) into your project:

```cpp
#include <NeuralNetworkCPP/DataPreprocessing/Scalers/MinMaxScaler/MinMaxScaler.hpp>
```

Create a scaler:

```cpp
nn::MinMaxScaler scaler;
```

Now you can fit date into the scaler:

```cpp
scaler.fit(mnistTrain.getData());
```

After fitting the data you can transform it to retrieve the normalized values:

```cpp
std::vector<std::vector<double>> scaledData = scaler.transform(mnistTrain.getData());
```

Or you can fit and transform the data at once using:

```cpp
std::vector<std::vector<double>> scaledData = scaler.fitTransform(mnistTrain.getData());
```

### [StandardScaler](docs/Classes/classnn_1_1_standard_scaler.md)

Start by including [StandardScaler](docs/Classes/classnn_1_1_standard_scaler.md) into your project:

```cpp
#include <NeuralNetworkCPP/DataPreprocessing/Scalers/StandardScaler/StandardScaler.hpp>
```

Create a scaler:

```cpp
nn::StandardScaler scaler;
```

Now you can fit date into the scaler:

```cpp
scaler.fit(trainData.getData());
```

After fitting the data you can transform it to retrieve the normalized values:

```cpp
std::vector<std::vector<double>> scaledData = scaler.transform(trainData.getData());
```

Or you can fit and transform the data at once using:

```cpp
std::vector<std::vector<double>> scaledData = scaler.fitTransform(trainData.getData());
```

### [to_categorical](docs/Namespaces/namespacenn.md#function-to_categorical)

Start by including [to_categorical](docs/Namespaces/namespacenn.md#function-to_categorical) helper function into your project:

```cpp
#include <NeuralNetworkCPP/Utils/Utils.hpp>
```

Converting a vector of class labels into one-hot encoded vectors:

```cpp
std::vector<std::vector<double>> trainLabels = nn::to_categorical(trainData.getLabels());
```

## Initializers

Two types of weights initialization were implemented in the project: [Xavier/Glorot](docs/Classes/classnn_1_1_xavier_uniform.md) and [He](docs/Classes/classnn_1_1_he_normal.md).

### Xavier/Glorot initializer

Xavier Uniform initialization is designed for networks using for example sigmoid or tanh activation functions. It draws weights from a uniform distribution within the range [-limit, limit], where limit = sqrt(6 / (fan-in + fan-out)), and fan-in and fan-out are the number of input and output neurons:

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(8, 1, nn::XAVIER_UNIFORM, nn::SIGMOID));
```

For flexibility Xavier Normal was also implemented in the project:

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(8, 1, nn::XAVIER_NORMAL, nn::SIGMOID));
```

### He initializer

He Normal initialization is designed for networks using [ReLU](docs/Classes/classnn_1_1_re_l_u.md) (or variants) activation functions. It draws weights from a normal distribution with a mean of 0 and a standard deviation of sqrt(2 / fan-in), where fan-in is the number of input neurons:

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(2, 8, nn::HE_NORMAL, nn::RELU));
```

For flexibility He Uniform was also implemented in the project:

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(2, 8, nn::HE_UNIFORM, nn::RELU));
```

## Activation functions

These are nonlinear functions that decide whether neurons should be activated i.e. pass information on, making the output of one neuron become the input of the next. Without them, a neural network, regardless of the number of layers, would be equivalent to a single linear model. The following activation functions were implemented in the project:
* [nn::RELU](docs/Classes/classnn_1_1_re_l_u.md)
* [nn::SIGMOID](docs/Classes/classnn_1_1_sigmoid.md)
* [nn::SOFTMAX](docs/Classes/classnn_1_1_softmax.md)
* nn::NONE (for linear regression)

### [ReLU](docs/Classes/classnn_1_1_re_l_u.md)

Used mainly in hidden layers:

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(2, 8, nn::HE_NORMAL, nn::RELU));
```

### [Sigmoid](docs/Classes/classnn_1_1_sigmoid.md)

Used mainly in binary classification problems as an activation function of the output layer:

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(8, 1, nn::XAVIER_UNIFORM, nn::SIGMOID));
```

### [Softmax](docs/Classes/classnn_1_1_softmax.md)

Used mainly in multi-class classification problems as an activation function of the output layer:

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(64, 10, nn::XAVIER_UNIFORM, nn::SOFTMAX));
```

### None

When used, it means that the layer should not apply any activation function. Used in linear regression problems for output layer:

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(26, 1, nn::XAVIER_UNIFORM, nn::NONE));
```

## Loss functions

Loss functions allow to calculate the error (difference) between predicted output and expected values. In this project, the following loss functions were implemented:
* [nn::BinaryCrossEntropy](docs/Classes/classnn_1_1_binary_cross_entropy.md)
* [nn::CategoricalCrossEntropy](docs/Classes/classnn_1_1_categorical_cross_entropy.md)
* [nn::MeanSquaredError](docs/Classes/classnn_1_1_mean_squared_error.md)

### [BinaryCrossEntropy](docs/Classes/classnn_1_1_binary_cross_entropy.md)

Binary Cross-Entropy is used for binary classification tasks. It measures the difference between predicted probabilities and true binary labels:

```cpp
model.compile(
    std::make_unique<nn::Adam>(),
    std::make_unique<nn::BinaryCrossEntropy>()
);
```

### [CategoricalCrossEntropy](docs/Classes/classnn_1_1_categorical_cross_entropy.md)

Categorical Cross-Entropy is used for multi-class classification tasks. It measures the difference between predicted class probabilities and true class labels:

```cpp
model.compile(
    std::make_unique<nn::Adam>(),
    std::make_unique<nn::CategoricalCrossEntropy>()
);
```

### [MeanSquaredError](docs/Classes/classnn_1_1_mean_squared_error.md)

MSE is commonly used for regression tasks. It measures the average squared difference between predicted and target values:

```cpp
model.compile(
    std::make_unique<nn::Adam>(),
    std::make_unique<nn::MeanSquaredError>()
);
```

## Layers

Two types of layers were implemented in the project:
* [nn::DenseLayer](docs/Classes/classnn_1_1_dense_layer.md)
* [nn::BatchNormalization](docs/Classes/classnn_1_1_batch_normalization.md)

### [DenseLayer](docs/Classes/classnn_1_1_dense_layer.md)

A fully connected (dense) layer applies a linear transformation (weights * input + biases). To create a dense layer the following must be provided:
* number of input neurons
* number of output neurons
* initializer
* activation function

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(28 * 28, 64, nn::HE_NORMAL, nn::RELU));
model.addLayer(std::make_unique<nn::DenseLayer>(64, 10, nn::XAVIER_UNIFORM, nn::SOFTMAX));
```

### [BatchNormalization](docs/Classes/classnn_1_1_batch_normalization.md)

Batch Normalization is a technique used to normalize the inputs of each layer to improve the training speed and stability of neural networks. It is important to use it only after each dense layer except for the output layer. To create this layer the number of features (or in simpler terms: number of output neurons of the previous dense layer) needs to be provided:

```cpp
model.addLayer(std::make_unique<nn::DenseLayer>(28 * 28, 64, nn::HE_NORMAL, nn::RELU));
model.addLayer(std::make_unique<nn::BatchNormalization>(64));
model.addLayer(std::make_unique<nn::DenseLayer>(64, 10, nn::XAVIER_UNIFORM, nn::SOFTMAX));
```
## Optimizers

Three optimizers were implemented in the project:
* [nn::SGD](docs/Classes/classnn_1_1_s_g_d.md) (with momentum)
* [nn::RMSprop](docs/Classes/classnn_1_1_r_m_sprop.md)
* [nn::Adam](docs/Classes/classnn_1_1_adam.md)

### [SGD](docs/Classes/classnn_1_1_s_g_d.md)

This optimizer updates parameters using the gradient of the loss function and a momentum term to accelerate convergence. When creating this optimizer it is possible to optionally provide:
* learning rate
* momentum

```cpp
model.compile(
    std::make_unique<nn::SGD>(0.001, 0.9),
    std::make_unique<nn::MeanSquaredError>()
);
```

### [RMSprop](docs/Classes/classnn_1_1_r_m_sprop.md)

This optimizer divides the learning rate by an exponentially decaying average of squared gradients to normalize the updates. When creating this optimizer it is possible to optionally provide:
* learning rate
* decay rate for the moving average
* small constant for numerical stability

```cpp
model.compile(
    std::make_unique<nn::RMSprop>(0.001, 0.9, 1e-8),
    std::make_unique<nn::MeanSquaredError>()
);
```

### [Adam](docs/Classes/classnn_1_1_adam.md)

This optimizer combines the benefits of momentum and [RMSprop](docs/Classes/classnn_1_1_r_m_sprop.md) to achieve faster convergence and better performance on a wide range of problems. When creating this optimizer it is possible to optionally provide:
* learning rate
* exponential decay rate for the first moment estimates
* exponential decay rate for the second moment estimates
* small constant for numerical stability

```cpp
model.compile(
    std::make_unique<nn::Adam>(0.001, 0.9, 0.999, 1e-8),
    std::make_unique<nn::MeanSquaredError>()
);
```
