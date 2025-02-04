/**
 * C++ neural network library
 *
 * ThreadPool.hpp
 */

#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <type_traits>

namespace nn
{
    class ThreadPool
    {
    private:
        std::vector<std::thread> m_workers;
        std::queue<std::function<void()>> m_tasks;
        std::mutex m_queueMutex;
        std::condition_variable m_condition;
        std::atomic<bool> m_stop;
        int m_numThreads;

    public:
        ThreadPool(int numThreads = std::thread::hardware_concurrency());

        ~ThreadPool();

        template <typename Func, typename... Args>
        auto enqueue(Func &&func, Args &&...args) -> std::future<typename std::invoke_result<Func, Args...>::type>;

        /**
         * @brief Executes a parallel for loop from `start` to `end` using multiple threads.
         *
         * @tparam _Func Function type (should accept an integer index).
         * @param start Start index.
         * @param end End index.
         * @param func Function to execute on each index.
         */
        template <typename Func>
        void parallelFor(int start, int end, Func func);
    };
}

#include "ThreadPool.tpp"

#endif