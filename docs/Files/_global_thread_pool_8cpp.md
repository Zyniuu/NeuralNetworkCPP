# GlobalThreadPool/GlobalThreadPool.cpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |




## Source code

```cpp


#include "GlobalThreadPool.hpp"

namespace nn
{
    // Initialize the global thread pool to nullptr (uninitialized by default).
    std::unique_ptr<ThreadPool> globalThreadPool = nullptr;

    void initGlobalThreadPool(int numThreads)
    {
        // Check if the global thread pool is already initialized.
        if (!globalThreadPool)
        {
            // Create a new ThreadPool instance with the specified number of threads.
            globalThreadPool = std::make_unique<ThreadPool>(numThreads);
        }
    }

    ThreadPool &getGlobalThreadPool()
    {
        // Throw an error if the thread pool is accessed before initialization.
        if (!globalThreadPool)
            throw std::runtime_error("Global thread pool not initialized.");

        // Return a reference to the global thread pool.
        return *globalThreadPool;
    }
}
```
