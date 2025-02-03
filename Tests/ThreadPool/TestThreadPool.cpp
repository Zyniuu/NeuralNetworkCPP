/**
 * C++ neural network library
 *
 * TestThreadPool.cpp
 */

#include <gtest/gtest.h>
#include <atomic>
#include "../../NeuralNetworkCPP/ThreadPool/ThreadPool.hpp"

// Test if threading is working correctly
TEST(ThreadPoolTests, ParallelSum)
{
    int N = 1000;
    std::atomic<int> sum{0};

    nn::ThreadPool::parallelFor(0, N, [&](int i) {
        sum.fetch_add(i, std::memory_order_relaxed);
    });

    int expected = (N * (N - 1)) / 2;
    EXPECT_EQ(sum.load(), expected);
}

// Test if function is executed correctly for small ranges
TEST(ThreadPoolTests, SmallRange)
{
    int counter = 0;

    nn::ThreadPool::parallelFor(0, 5, [&](int) {
        counter++;
    });

    EXPECT_EQ(counter, 5);
}

// Test if empty ranges are ignored
TEST(ThreadPoolTests, EdgeCaseEmptyRange)
{
    int counter = 0;

    nn::ThreadPool::parallelFor(5, 5, [&](int) {
        counter++;
    });

    EXPECT_EQ(counter, 0);
}