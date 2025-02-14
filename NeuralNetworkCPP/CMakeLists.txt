# Minimal version of CMake required
cmake_minimum_required(VERSION 3.10)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Project name
project(NeuralNetworCPP)

# List of source files of the neural network
set(NN_SOURCES
    Utils/Utils.cpp
    DataPreprocessing/CSVReader/CSVReader.cpp
    DataPreprocessing/Scalers/StandardScaler/StandardScaler.cpp
    Logger/Logger.cpp
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
    Layers/BatchNormalization/BatchNormalization.cpp
    NeuralNetworkCPP.cpp
)

# Create static library
add_library(${PROJECT_NAME} STATIC ${NN_SOURCES})

# Add header directories
target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Set output directories
set_target_properties(${PROJECT_NAME} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ../lib
    LIBRARY_OUTPUT_DIRECTORY ../lib
)
