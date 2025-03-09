# nn::StandardScaler



Normalizes data to have a mean of 0 and a standard deviation of 1. 


`#include <StandardScaler.hpp>`

Inherits from [nn::Scaler](classnn_1_1_scaler.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| virtual void | **[fit](classnn_1_1_standard_scaler.md#function-fit)**(const std::vector< std::vector< double > > & data) override<br>Fits the scaler to the data (computes mean and standard deviation).  |
| virtual std::vector< std::vector< double > > | **[transform](classnn_1_1_standard_scaler.md#function-transform)**(const std::vector< std::vector< double > > & data) override<br>Transforms the data using the computed mean and standard deviation.  |
| virtual std::vector< std::vector< double > > | **[fitTransform](classnn_1_1_standard_scaler.md#function-fittransform)**(const std::vector< std::vector< double > > & data) override<br>Fits the scaler to the data and then transforms the data.  |

## Public Functions Documentation

### function fit

```cpp
virtual void fit(
    const std::vector< std::vector< double > > & data
) override
```

Fits the scaler to the data (computes mean and standard deviation). 

**Parameters**: 

  * **data** The input data as a vector of vectors of doubles. 


**Reimplements**: [nn::Scaler::fit](classnn_1_1_scaler.md#function-fit)


### function transform

```cpp
virtual std::vector< std::vector< double > > transform(
    const std::vector< std::vector< double > > & data
) override
```

Transforms the data using the computed mean and standard deviation. 

**Parameters**: 

  * **data** The input data as a vector of vectors of doubles. 


**Return**: std::vector<std::vector<double>> The normalized data. 

**Reimplements**: [nn::Scaler::transform](classnn_1_1_scaler.md#function-transform)


### function fitTransform

```cpp
virtual std::vector< std::vector< double > > fitTransform(
    const std::vector< std::vector< double > > & data
) override
```

Fits the scaler to the data and then transforms the data. 

**Parameters**: 

  * **data** The input data as a vector of vectors of doubles. 


**Return**: std::vector<std::vector<double>> The normalized data. 

**Reimplements**: [nn::Scaler::fitTransform](classnn_1_1_scaler.md#function-fittransform)
