# nn::MinMaxScaler



Normalizes data to a specified range (default: [0, 1]). 


`#include <MinMaxScaler.hpp>`

Inherits from [nn::Scaler](classnn_1_1_scaler.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[MinMaxScaler](classnn_1_1_min_max_scaler.md#function-minmaxscaler)**(const double featureRangeMin =0.0, const double featureRangeMax =1.0)<br>Constructs a [MinMaxScaler](classnn_1_1_min_max_scaler.md) object.  |
| virtual void | **[fit](classnn_1_1_min_max_scaler.md#function-fit)**(const std::vector< std::vector< double > > & data) override<br>Fits the scaler to the data (computes min and max).  |
| virtual std::vector< std::vector< double > > | **[transform](classnn_1_1_min_max_scaler.md#function-transform)**(const std::vector< std::vector< double > > & data) override<br>Transforms the data using the computed min and max.  |
| virtual std::vector< std::vector< double > > | **[fitTransform](classnn_1_1_min_max_scaler.md#function-fittransform)**(const std::vector< std::vector< double > > & data) override<br>Fits the scaler to the data and then transforms the data.  |

## Public Functions Documentation

### function MinMaxScaler

```cpp
MinMaxScaler(
    const double featureRangeMin =0.0,
    const double featureRangeMax =1.0
)
```

Constructs a [MinMaxScaler](classnn_1_1_min_max_scaler.md) object. 

**Parameters**: 

  * **featureRangeMin** The minimum value of the target range (default: 0.0). 
  * **featureRangeMax** The maximum value of the target range (default: 1.0). 


### function fit

```cpp
virtual void fit(
    const std::vector< std::vector< double > > & data
) override
```

Fits the scaler to the data (computes min and max). 

**Parameters**: 

  * **data** The input data as a vector of vectors of doubles. 


**Reimplements**: [nn::Scaler::fit](classnn_1_1_scaler.md#function-fit)


### function transform

```cpp
virtual std::vector< std::vector< double > > transform(
    const std::vector< std::vector< double > > & data
) override
```

Transforms the data using the computed min and max. 

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
