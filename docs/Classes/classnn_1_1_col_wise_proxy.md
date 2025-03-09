# nn::ColWiseProxy



Proxy class for performing column-wise operations on a matrix.  [More...](#detailed-description)


`#include <ColWiseProxy.hpp>`

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[ColWiseProxy](classnn_1_1_col_wise_proxy.md#function-colwiseproxy)**(const [Matrix](classnn_1_1_matrix.md) & matrix)<br>Constructs a [ColWiseProxy](classnn_1_1_col_wise_proxy.md) for the given matrix.  |
| [Matrix](classnn_1_1_matrix.md) | **[maxCoeff](classnn_1_1_col_wise_proxy.md#function-maxcoeff)**() const<br>Returns a row vector with maximum coefficient of each column.  |
| [Matrix](classnn_1_1_matrix.md) | **[sum](classnn_1_1_col_wise_proxy.md#function-sum)**() const<br>Sums elements in each column.  |

## Friends

|                | Name           |
| -------------- | -------------- |
| [Matrix](classnn_1_1_matrix.md) | **[operator*](classnn_1_1_col_wise_proxy.md#friend-operator*)**(const [ColWiseProxy](classnn_1_1_col_wise_proxy.md) & left, const [Matrix](classnn_1_1_matrix.md) & right) <br>Multiplies a column vector by every column of a matrix.  |
| [Matrix](classnn_1_1_matrix.md) | **[operator/](classnn_1_1_col_wise_proxy.md#friend-operator/)**(const [ColWiseProxy](classnn_1_1_col_wise_proxy.md) & left, const [Matrix](classnn_1_1_matrix.md) & right) <br>Divides every column of a matrix by a column vector.  |
| [Matrix](classnn_1_1_matrix.md) | **[operator+](classnn_1_1_col_wise_proxy.md#friend-operator+)**(const [ColWiseProxy](classnn_1_1_col_wise_proxy.md) & left, const [Matrix](classnn_1_1_matrix.md) & right) <br>Adds a column vector to every column of a matrix.  |
| [Matrix](classnn_1_1_matrix.md) | **[operator-](classnn_1_1_col_wise_proxy.md#friend-operator-)**(const [ColWiseProxy](classnn_1_1_col_wise_proxy.md) & left, const [Matrix](classnn_1_1_matrix.md) & right) <br>Subtracts a column vector from every column of a matrix.  |

## Detailed Description

```cpp
class nn::ColWiseProxy;
```

Proxy class for performing column-wise operations on a matrix. 

This class enables operations like multiplying a column vector by all columns of a matrix. It is returned by the [Matrix::colWise()](classnn_1_1_matrix.md#function-colwise) method. 

## Public Functions Documentation

### function ColWiseProxy

```cpp
ColWiseProxy(
    const Matrix & matrix
)
```

Constructs a [ColWiseProxy](classnn_1_1_col_wise_proxy.md) for the given matrix. 

**Parameters**: 

  * **matrix** The matrix to perform column-wise operations on. 


### function maxCoeff

```cpp
Matrix maxCoeff() const
```

Returns a row vector with maximum coefficient of each column. 

**Return**: The row vector with maximum coefficient of each column. 

### function sum

```cpp
Matrix sum() const
```

Sums elements in each column. 

**Return**: Row matrix with summed up columns. 

## Friends

### friend operator*

```cpp
friend Matrix operator*(
    const ColWiseProxy & left,

    const Matrix & right
);
```

Multiplies a column vector by every column of a matrix. 

**Parameters**: 

  * **other** A column vector ([Matrix](classnn_1_1_matrix.md) with 1 column). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a column vector or its row count does not match the original matrix. 


**Return**: A new [Matrix](classnn_1_1_matrix.md) after the column-wise multiplication. 

### friend operator/

```cpp
friend Matrix operator/(
    const ColWiseProxy & left,

    const Matrix & right
);
```

Divides every column of a matrix by a column vector. 

**Parameters**: 

  * **other** A column vector ([Matrix](classnn_1_1_matrix.md) with 1 column). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a column vector or its row count does not match the original matrix. 


**Return**: A new [Matrix](classnn_1_1_matrix.md) after the column-wise division. 

### friend operator+

```cpp
friend Matrix operator+(
    const ColWiseProxy & left,

    const Matrix & right
);
```

Adds a column vector to every column of a matrix. 

**Parameters**: 

  * **other** A column vector ([Matrix](classnn_1_1_matrix.md) with 1 column). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a column vector or its row count does not match the original matrix. 


**Return**: A new [Matrix](classnn_1_1_matrix.md) after the column-wise addition. 

### friend operator-

```cpp
friend Matrix operator-(
    const ColWiseProxy & left,

    const Matrix & right
);
```

Subtracts a column vector from every column of a matrix. 

**Parameters**: 

  * **other** A column vector ([Matrix](classnn_1_1_matrix.md) with 1 column). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a column vector or its row count does not match the original matrix. 


**Return**: A new [Matrix](classnn_1_1_matrix.md) after the column-wise subtraction. 
