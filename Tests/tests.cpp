/**
 * C++ neural network library
 *
 * tests.cpp
 */

#include <gtest/gtest.h>
#include "../NeuralNetworkCPP/GlobalThreadPool/GlobalThreadPool.hpp"

int main(int argc, char **argv)
{
    nn::initGlobalThreadPool();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}