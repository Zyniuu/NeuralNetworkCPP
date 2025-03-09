# nn::ThreadPool



A thread pool implementation for executing tasks in parallel.  [More...](#detailed-description)


`#include <ThreadPool.hpp>`

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[ThreadPool](classnn_1_1_thread_pool.md#function-threadpool)**(int numThreads)<br>Constructs a [ThreadPool](classnn_1_1_thread_pool.md) with the specified number of threads.  |
| | **[~ThreadPool](classnn_1_1_thread_pool.md#function-~threadpool)**()<br>Destructor. Stops the thread pool and joins all worker threads.  |
| int | **[getThreadCount](classnn_1_1_thread_pool.md#function-getthreadcount)**() const<br>Retrieves the thread count.  |
| template <typename Func ,typename... Args\> <br>std::future< typename std::invoke_result< Func, Args... >::type > | **[enqueue](classnn_1_1_thread_pool.md#function-enqueue)**(Func && func, Args &&... args)<br>Enqueues a task to be executed by the thread pool.  |
| template <typename Func \> <br>void | **[parallelFor](classnn_1_1_thread_pool.md#function-parallelfor)**(int start, int end, Func func)<br>Executes a parallel for loop from `start` to `end` using multiple threads.  |

## Detailed Description

```cpp
class nn::ThreadPool;
```

A thread pool implementation for executing tasks in parallel. 

This class manages a pool of worker threads that execute tasks from a queue. Tasks can be enqueued using the `enqueue` method, and parallel loops can be executed using the `parallelFor` method. 

## Public Functions Documentation

### function ThreadPool

```cpp
ThreadPool(
    int numThreads
)
```

Constructs a [ThreadPool](classnn_1_1_thread_pool.md) with the specified number of threads. 

**Parameters**: 

  * **numThreads** The number of threads in the pool. 


### function ~ThreadPool

```cpp
~ThreadPool()
```

Destructor. Stops the thread pool and joins all worker threads. 

### function getThreadCount

```cpp
int getThreadCount() const
```

Retrieves the thread count. 

**Return**: The thread count. 

### function enqueue

```cpp
template <typename Func ,
typename... Args>
std::future< typename std::invoke_result< Func, Args... >::type > enqueue(
    Func && func,
    Args &&... args
)
```

Enqueues a task to be executed by the thread pool. 

**Parameters**: 

  * **func** The callable object to execute. 
  * **args** The arguments to pass to the callable. 


**Template Parameters**: 

  * **Func** The type of the callable object (function, lambda, etc.). 
  * **Args** The types of the arguments to pass to the callable. 


**Return**: A std::future representing the result of the task. 

### function parallelFor

```cpp
template <typename Func >
void parallelFor(
    int start,
    int end,
    Func func
)
```

Executes a parallel for loop from `start` to `end` using multiple threads. 

**Parameters**: 

  * **start** The start index (inclusive). 
  * **end** The end index (exclusive). 
  * **func** The function to execute on each index. 


**Template Parameters**: 

  * **Func** The type of the function to execute on each index. 
