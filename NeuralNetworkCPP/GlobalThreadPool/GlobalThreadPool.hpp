/**
 * C++ neural network library
 *
 * GlobalThreadPool.hpp
 */

#ifndef GLOBALTHREADPOOL_HPP
#define GLOBALTHREADPOOL_HPP

#include "Base/ThreadPool.hpp"
#include <memory>

namespace nn
{
    /**
     * @brief Global thread pool instance accessible throughout the program.
     *
     * This is a singleton-like thread pool to avoid redundant thread creation
     * and ensure efficient resource usage across the application.
     */
    extern std::unique_ptr<ThreadPool> globalThreadPool;

    /**
     * @brief Initializes the global thread pool with the specified number of threads.
     *
     * @param numThreads The number of threads to create. Defaults to the number of hardware threads.
     */
    void initGlobalThreadPool(int numThreads = std::thread::hardware_concurrency());

    /**
     * @brief Returns the global thread pool instance.
     *
     * @return A reference to the global thread pool.
     * @throws std::runtime_error If the thread pool is not initialized.
     */
    ThreadPool &getGlobalThreadPool();
}

#endif