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
    /**
     * @class ThreadPool
     * @brief A thread pool implementation for executing tasks in parallel.
     *
     * This class manages a pool of worker threads that execute tasks from a queue.
     * Tasks can be enqueued using the `enqueue` method, and parallel loops can be
     * executed using the `parallelFor` method.
     */
    class ThreadPool
    {
    private:
        std::vector<std::thread> m_workers;        ///< Worker threads in the pool.
        std::queue<std::function<void()>> m_tasks; ///< Queue of tasks to execute.
        std::mutex m_queueMutex;                   ///< Mutex to protect access to the task queue.
        std::condition_variable m_condition;       ///< Condition variable for task synchronization.
        std::atomic<bool> m_stop;                  ///< Flag to indicate if the thread pool should stop.
        int m_numThreads;                          ///< Number of threads in the pool.

    public:
        /**
         * @brief Constructs a ThreadPool with the specified number of threads.
         *
         * @param numThreads The number of threads in the pool.
         */
        ThreadPool(int numThreads);

        /**
         * @brief Destructor. Stops the thread pool and joins all worker threads.
         */
        ~ThreadPool();

        /**
         * @brief Enqueues a task to be executed by the thread pool.
         *
         * @tparam Func The type of the callable object (function, lambda, etc.).
         * @tparam Args The types of the arguments to pass to the callable.
         * @param func The callable object to execute.
         * @param args The arguments to pass to the callable.
         * @return A std::future representing the result of the task.
         */
        template <typename Func, typename... Args>
        auto enqueue(Func &&func, Args &&...args) -> std::future<typename std::invoke_result<Func, Args...>::type>;

        /**
         * @brief Executes a parallel for loop from `start` to `end` using multiple threads.
         *
         * @tparam Func The type of the function to execute on each index.
         * @param start The start index (inclusive).
         * @param end The end index (exclusive).
         * @param func The function to execute on each index.
         */
        template <typename Func>
        void parallelFor(int start, int end, Func func);
    };
}

#include "ThreadPool.tpp"

#endif