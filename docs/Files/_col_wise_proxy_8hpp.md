# Matrix/ColWiseProxy/ColWiseProxy.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::ColWiseProxy](../Classes/classnn_1_1_col_wise_proxy.md)** <br>Proxy class for performing column-wise operations on a matrix.  |




## Source code

```cpp


#ifndef COLWISEPROXY_HPP
#define COLWISEPROXY_HPP

#include "../Matrix.hpp"

namespace nn
{
    // Forward declaration of Matrix class
    class Matrix;

    class ColWiseProxy
    {
    private:
        const Matrix &m_matrix; 

    public:
        ColWiseProxy(const Matrix &matrix);

        Matrix maxCoeff() const;

        Matrix sum() const;

        friend Matrix operator*(const ColWiseProxy &left, const Matrix &right);

        friend Matrix operator/(const ColWiseProxy &left, const Matrix &right);

        friend Matrix operator+(const ColWiseProxy &left, const Matrix &right);

        friend Matrix operator-(const ColWiseProxy &left, const Matrix &right);
    };
}

#endif
```
