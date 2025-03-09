# GlobalThreadPool/Base/ThreadPool.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::ThreadPool](../Classes/classnn_1_1_thread_pool.md)** <br>A thread pool implementation for executing tasks in parallel.  |




## Source code

```cpp


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
        ThreadPool(int numThreads);

        ~ThreadPool();

        int getThreadCount() const;

        template <typename Func, typename... Args>
        auto enqueue(Func &&func, Args &&...args) -> std::future<typename std::invoke_result<Func, Args...>::type>;

        template <typename Func>
        void parallelFor(int start, int end, Func func);
    };
}

#include "ThreadPool.tpp"

#endif
```
