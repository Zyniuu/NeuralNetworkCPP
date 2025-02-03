/**
 * C++ neural network library
 *
 * ThreadPool.tpp
 */

#ifndef THREADPOOL_TPP
#define THREADPOOL_TPP

#include "ThreadPool.hpp"
#include <thread>
#include <vector>
#include <future>
#include <algorithm>

namespace nn
{
    template <typename _Func>
    void ThreadPool::parallelFor(int start, int end, _Func func)
    {
        // Determine the number of threads to use
        int numThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        int rangeSize = end - start;

        // If the range is small or there's only one thread, execute sequentially
        if (rangeSize < numThreads || numThreads <= 1)
        {
            for (int i = start; i < end; i++)
                func(i);
            return;
        }

        // Split the work into chunks
        std::vector<std::future<void>> futures;
        int chunkSize = (rangeSize + numThreads - 1) / numThreads; // Round up

        // Launch threads to process each chunk
        for (int i = 0; i < numThreads; i++)
        {
            int chunkStart = start + i * chunkSize;
            int chunkEnd = std::min(chunkStart + chunkSize, end);

            if (chunkStart < chunkEnd) // Avoid empty ranges
            {
                futures.push_back(std::async(std::launch::async, [=]() {
                    for (int j = chunkStart; j < chunkEnd; j++)
                        func(j);
                }));
            }
        }

        // Wait for all threads to finish
        for (auto &f : futures)
            f.get();
    }
}

#endif