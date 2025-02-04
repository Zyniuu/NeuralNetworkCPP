/**
 * C++ neural network library
 *
 * GlobalThreadPool.cpp
 */

#include "GlobalThreadPool.hpp"

namespace nn
{
    std::unique_ptr<ThreadPool> globalThreadPool = nullptr;

    void initGlobalThreadPool(int numThreads)
    {
        if (!globalThreadPool)
            globalThreadPool = std::make_unique<ThreadPool>(numThreads);
    }

    ThreadPool &getGlobalThreadPool()
    {
        if (!globalThreadPool)
            throw std::runtime_error("Global thread pool not initialized.");
        return *globalThreadPool;
    }
}