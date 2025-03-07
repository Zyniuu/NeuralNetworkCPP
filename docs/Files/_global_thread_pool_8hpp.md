# GlobalThreadPool/GlobalThreadPool.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |




## Source code

```cpp


#ifndef GLOBALTHREADPOOL_HPP
#define GLOBALTHREADPOOL_HPP

#include "Base/ThreadPool.hpp"
#include <memory>

namespace nn
{
    extern std::unique_ptr<ThreadPool> globalThreadPool;

    void initGlobalThreadPool(int numThreads = std::thread::hardware_concurrency());

    ThreadPool &getGlobalThreadPool();
}

#endif
```
