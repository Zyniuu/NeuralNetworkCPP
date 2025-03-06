# NeuralNetworkCPP
A lightweight, modular, and easy-to-use C++ neural network library designed for building, training, and evaluating neural networks. The library implements features such as: batch processing, early stopping and multi threading.

# Table of contents

* [Usage](#usage)
* [Data preprocessing](#data-preprocessing)
    * [CSVReader](#csvreader)
    * [MinMaxScaler](#minmaxscaler)
    * [StandardScaler](#standardscaler)
* [Initializers](#initializers)
    * [Xavier/Glorot initializer](#xavierglorot-initializer)
    * [He initializer](#he-initializer)
* [Activation functions](#activation-functions)
    * [ReLU](#relu)
    * [Sigmoid](#sigmoid)
    * [Softmax](#sofmtax)
* [Loss functions](#loss-functions)
    * [BinaryCrossEntropy](#binarycrossentropy)
    * [CategoricalCrossEntropy](#categoricalcrossentropy)
    * [MeanSquaredError](#meansquarederror)
* [Layers](#layers)
    * [DenseLayer](#denselayer)
    * [BatchNormalization](#batchnormalization)
* [Optimizers](#optimizers)
    * [Adam](#adam)
    * [RMSprop](#rmsprop)
    * [SGD](#sgd)

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

