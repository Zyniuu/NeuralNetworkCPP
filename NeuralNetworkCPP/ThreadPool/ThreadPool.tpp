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

namespace nn
{
    template <typename _Func>
    inline void ThreadPool::parallelFor(const int &start, const int &end, _Func func)
    {
        int numThreads = std::thread::hardware_concurrency();

        if (numThreads <= 1 || (end - start) < 10) // Avoid unnecessary threading
        {
            for (int i = start; i < end; i++)
                func(i);
            return;
        }

        std::vector<std::future<void>> futures;
        int chunkSize = (end - start + numThreads - 1) / numThreads; // Round up

        for (int i = 0; i < numThreads; i++)
        {
            int chunkStart = start + i * chunkSize;
            int chunkEnd = std::min(chunkStart + chunkSize, end);

            if (chunkStart < chunkEnd) // Avoid empty ranges
            {
                futures.push_back(
                    std::async(
                        std::launch::async, [=]()
                        {
                            for (int j = chunkStart; j < chunkEnd; j++)
                                func(j);
                        }
                    )
                );
            }
        }

        for (auto &f : futures)
            f.get();
    }
}

#endif