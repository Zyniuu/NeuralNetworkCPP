/**
 * C++ neural network library
 *
 * ThreadPool.hpp
 */

#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

namespace nn
{
    class ThreadPool
    {
    public:
        /**
         * @brief Executes a parallel for loop from `start` to `end` using multiple threads.
         * 
         * @tparam _Func Function type (should accept an integer index).
         * @param start Start index (inclusive).
         * @param end End index (exclusive).
         * @param func Function to execute on each index.
         */
        template <typename _Func>
        static inline void parallelFor(const int &start, const int &end, _Func func);
    };
}

#include "ThreadPool.tpp"

#endif