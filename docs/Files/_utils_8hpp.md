# Utils/Utils.hpp

This file contains helper functions.

## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Source code

```cpp


#ifndef UTILS_HPP
#define UTILS_HPP


#include <vector>

namespace nn
{
    std::vector<std::vector<double>> slice(const std::vector<std::vector<double>> &data, const int start, const int end);

    void reorderRows(std::vector<std::vector<double>> &data, const std::vector<int> &order);

    void shuffleDataset(std::vector<std::vector<double>> &data, std::vector<std::vector<double>> &labels);

    std::vector<std::vector<double>> to_categorical(const std::vector<std::vector<double>> &data, int numClasses = 0);
}

#endif
```
