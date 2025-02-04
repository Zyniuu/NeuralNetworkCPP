/**
 * C++ neural network library
 *
 * ThreadPool.cpp
 */

#include "ThreadPool.hpp"
#include <utility>

namespace nn
{
    ThreadPool::ThreadPool(int numThreads)
        : m_stop(false), m_numThreads(numThreads)
    {
        // Create worker threads.
        for (int i = 0; i < numThreads; i++)
        {
            m_workers.emplace_back([this] {
                while (true)
                {
                    std::function<void()> task;

                    {
                        // Lock the mutex to safely access the task queue.
                        std::unique_lock<std::mutex> lock(this->m_queueMutex);

                        // Wait for a task to be available or for the pool to stop.
                        this->m_condition.wait(lock, [this] {
                            return this->m_stop || !this->m_tasks.empty();
                        });

                        // If the pool is stopped and no tasks remain, exit the thread.
                        if (this->m_stop && this->m_tasks.empty())
                            return;

                        // Get the next task from the queue.
                        task = std::move(this->m_tasks.front());
                        this->m_tasks.pop();
                    }

                    // Execute the task.
                    task();
                }
            });
        }
    }

    ThreadPool::~ThreadPool()
    {
        // Set the stop flag to true.
        m_stop = true;

        // Notify all worker threads to wake up.
        m_condition.notify_all();

        // Join all worker threads to ensure they finish execution.
        for (std::thread &worker : m_workers)
        {
            if (worker.joinable())
                worker.join();
        }
    }
}