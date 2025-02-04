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
        for (int i = 0; i < numThreads; i++)
        {
            m_workers.emplace_back([this] {
                while (true)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->m_queueMutex);
                        this->m_condition.wait(lock, [this] {
                            return this->m_stop || !this->m_tasks.empty();
                        });

                        if (this->m_stop && this->m_tasks.empty())
                            return;

                        task = std::move(this->m_tasks.front());
                        this->m_tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    ThreadPool::~ThreadPool()
    {
        m_stop = true;
        m_condition.notify_all();
        for (std::thread &worker : m_workers)
        {
            if (worker.joinable())
                worker.join();
        }
    }
}