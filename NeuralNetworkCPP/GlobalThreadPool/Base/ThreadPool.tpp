/**
 * C++ neural network library
 *
 * ThreadPool.tpp
 */

#ifndef THREADPOOL_TPP
#define THREADPOOL_TPP

#include "ThreadPool.hpp"
#include <algorithm>
#include <memory>

namespace nn
{
    template <typename Func, typename... Args>
    auto ThreadPool::enqueue(Func &&func, Args &&...args) -> std::future<typename std::invoke_result<Func, Args...>::type>
    {
        // Determine the return type of the callable.
        using returnType = typename std::invoke_result<Func, Args...>::type;

        // Create a packaged_task to wrap the callable and its arguments.
        auto task = std::make_shared<std::packaged_task<returnType()>>(
            std::bind(std::forward<Func>(func), std::forward<Args>(args)...));

        // Get a future to retrieve the result of the task.
        std::future<returnType> res = task->get_future();

        {
            // Lock the mutex to safely access the task queue.
            std::unique_lock<std::mutex> lock(m_queueMutex);

            // Throw an exception if the thread pool is stopped.
            if (m_stop)
                throw std::runtime_error("enqueue on stopped ThreadPool.");

            // Add the task to the queue.
            m_tasks.emplace([task]() {
                (*task)(); // Execute the task.
            });
        }

        // Notify one worker thread that a new task is available.
        m_condition.notify_one();

        // Return the future to the caller.
        return res;
    }

    template <typename Func>
    void ThreadPool::parallelFor(int start, int end, Func func)
    {
        // Determine the number of threads to use
        int numThreads = std::max(1, m_numThreads);
        int rangeSize = end - start;

        // If the range is small or there's only one thread, execute sequentially
        if (rangeSize < numThreads * 2 || numThreads <= 1)
        {
            for (int i = start; i < end; i++)
                func(i);
            return;
        }

        // Split the work into chunks
        int chunkSize = (rangeSize + numThreads - 1) / numThreads; // Round up
        std::vector<std::future<void>> futures;

        // Launch threads to process each chunk
        for (int i = 0; i < numThreads; i++)
        {
            int chunkStart = start + i * chunkSize;
            int chunkEnd = std::min(chunkStart + chunkSize, end);

            if (chunkStart < chunkEnd) // Avoid empty ranges
            {
                futures.emplace_back(enqueue([=] {
                    for (int j = chunkStart; j < chunkEnd; j++)
                        func(j);
                }));
            }
        }

        // Wait for all threads to finish
        for (auto &f : futures)
            f.wait();
    }
}

#endif