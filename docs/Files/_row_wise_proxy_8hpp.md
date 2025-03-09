# Matrix/RowWiseProxy/RowWiseProxy.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::RowWiseProxy](../Classes/classnn_1_1_row_wise_proxy.md)** <br>Proxy class for performing row-wise operations on a matrix.  |




## Source code

```cpp


#ifndef ROWWISEPROXY_HPP
#define ROWWISEPROXY_HPP

#include "../Matrix.hpp"

namespace nn
{
    // Forward declaration of Matrix class
    class Matrix;

    class RowWiseProxy
    {
    private:
        const Matrix &m_matrix; 

    public:
        RowWiseProxy(const Matrix &matrix);

        Matrix sum() const;

        friend Matrix operator-(const RowWiseProxy &left, const Matrix &right);

        friend Matrix operator/(const RowWiseProxy &left, const Matrix &right);
    };
}

#endif
```
