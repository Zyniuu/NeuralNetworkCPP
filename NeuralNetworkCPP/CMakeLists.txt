# Minimal version of CMake required
cmake_minimum_required(VERSION 3.10)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Project name
project(NeuralNetworCPP)

# List of source files of the neural network
set(NN_SOURCES
    Initializers/XavierNormal/XavierNormal.cpp
    Initializers/XavierUniform/XavierUniform.cpp
    Initializers/HeNormal/HeNormal.cpp
    Initializers/HeUniform/HeUniform.cpp
    GlobalThreadPool/Base/ThreadPool.tpp
    GlobalThreadPool/Base/ThreadPool.cpp
    GlobalThreadPool/GlobalThreadPool.cpp
    Matrix/Matrix.cpp
    Matrix/RowWiseProxy/RowWiseProxy.cpp
    Losses/MeanSquaredError/MeanSquaredError.cpp
    Losses/CategoricalCrossEntropy/CategoricalCrossEntropy.cpp
    Losses/BinaryCrossEntropy/BinaryCrossEntropy.cpp
    Optimizers/SGD/SGD.cpp
    Optimizers/RMSprop/RMSprop.cpp
    Optimizers/Adam/Adam.cpp
    Activations/ReLU/ReLU.cpp
    Activations/Sigmoid/Sigmoid.cpp
    Activations/Softmax/Softmax.cpp
    Layers/DenseLayer/DenseLayer.cpp
)

# Include directories for headers
set(NN_HEADERS
    Initializers
    GlobalThreadPool
    GlobalThreadPool/Base
    Matrix
    Matrix/RowWiseProxy
    Losses
    Optimizers
    Activations
    Layers
)

# Create static library
add_library(${PROJECT_NAME} STATIC ${NN_SOURCES})

# Add header directories
target_include_directories(${PROJECT_NAME} PUBLIC ${NN_HEADERS})

# Set output directories
set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ../lib
    LIBRARY_OUTPUT_DIRECTORY ../lib
)
