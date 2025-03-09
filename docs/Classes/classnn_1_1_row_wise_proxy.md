# nn::RowWiseProxy



Proxy class for performing row-wise operations on a matrix.  [More...](#detailed-description)


`#include <RowWiseProxy.hpp>`

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[RowWiseProxy](classnn_1_1_row_wise_proxy.md#function-rowwiseproxy)**(const [Matrix](classnn_1_1_matrix.md) & matrix)<br>Constructs a [RowWiseProxy](classnn_1_1_row_wise_proxy.md) for the given matrix.  |
| [Matrix](classnn_1_1_matrix.md) | **[sum](classnn_1_1_row_wise_proxy.md#function-sum)**() const<br>Sums elements in each row.  |

## Friends

|                | Name           |
| -------------- | -------------- |
| [Matrix](classnn_1_1_matrix.md) | **[operator-](classnn_1_1_row_wise_proxy.md#friend-operator-)**(const [RowWiseProxy](classnn_1_1_row_wise_proxy.md) & left, const [Matrix](classnn_1_1_matrix.md) & right) <br>Subtracts a row vector from each row of the matrix.  |
| [Matrix](classnn_1_1_matrix.md) | **[operator/](classnn_1_1_row_wise_proxy.md#friend-operator/)**(const [RowWiseProxy](classnn_1_1_row_wise_proxy.md) & left, const [Matrix](classnn_1_1_matrix.md) & right) <br>Divides a row vector by each row of the matrix.  |

## Detailed Description

```cpp
class nn::RowWiseProxy;
```

Proxy class for performing row-wise operations on a matrix. 

This class enables operations like adding a row vector to all rows of a matrix. It is returned by the [Matrix::rowWise()](classnn_1_1_matrix.md#function-rowwise) method. 

## Public Functions Documentation

### function RowWiseProxy

```cpp
RowWiseProxy(
    const Matrix & matrix
)
```

Constructs a [RowWiseProxy](classnn_1_1_row_wise_proxy.md) for the given matrix. 

**Parameters**: 

  * **matrix** The matrix to perform row-wise operations on. 


### function sum

```cpp
Matrix sum() const
```

Sums elements in each row. 

**Return**: Column matrix with summed up rows. 

## Friends

### friend operator-

```cpp
friend Matrix operator-(
    const RowWiseProxy & left,

    const Matrix & right
);
```

Subtracts a row vector from each row of the matrix. 

**Parameters**: 

  * **other** A row vector ([Matrix](classnn_1_1_matrix.md) with 1 row). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a row vector or its column count does not match the original matrix. 


**Return**: A new [Matrix](classnn_1_1_matrix.md) after the row-wise subtraction. 

### friend operator/

```cpp
friend Matrix operator/(
    const RowWiseProxy & left,

    const Matrix & right
);
```

Divides a row vector by each row of the matrix. 

**Parameters**: 

  * **other** A row vector ([Matrix](classnn_1_1_matrix.md) with 1 row). 


**Exceptions**: 

  * **std::invalid_argument** If `other` is not a row vector or its column count does not match the original matrix. 


**Return**: A new [Matrix](classnn_1_1_matrix.md) after the row-wise division. 
