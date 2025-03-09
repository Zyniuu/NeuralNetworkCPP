# nn::Scaler



Abstract base class for all scalers. 


`#include <Scaler.hpp>`

Inherited by [nn::MinMaxScaler](classnn_1_1_min_max_scaler.md), [nn::StandardScaler](classnn_1_1_standard_scaler.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| virtual void | **[fit](classnn_1_1_scaler.md#function-fit)**(const std::vector< std::vector< double > > & data) =0<br>Fits the scaler to the data.  |
| virtual std::vector< std::vector< double > > | **[transform](classnn_1_1_scaler.md#function-transform)**(const std::vector< std::vector< double > > & data) =0<br>Transforms the data using the fitted parameters.  |
| virtual std::vector< std::vector< double > > | **[fitTransform](classnn_1_1_scaler.md#function-fittransform)**(const std::vector< std::vector< double > > & data) =0<br>Fits the scaler to the data and then transforms the data.  |

## Public Functions Documentation

### function fit

```cpp
virtual void fit(
    const std::vector< std::vector< double > > & data
) =0
```

Fits the scaler to the data. 

**Parameters**: 

  * **data** The input data as a vector of vectors of doubles. 


**Reimplemented by**: [nn::MinMaxScaler::fit](classnn_1_1_min_max_scaler.md#function-fit), [nn::StandardScaler::fit](classnn_1_1_standard_scaler.md#function-fit)


### function transform

```cpp
virtual std::vector< std::vector< double > > transform(
    const std::vector< std::vector< double > > & data
) =0
```

Transforms the data using the fitted parameters. 

**Parameters**: 

  * **data** The input data as a vector of vectors of doubles. 


**Return**: std::vector<std::vector<double>> The normalized data. 

**Reimplemented by**: [nn::MinMaxScaler::transform](classnn_1_1_min_max_scaler.md#function-transform), [nn::StandardScaler::transform](classnn_1_1_standard_scaler.md#function-transform)


### function fitTransform

```cpp
virtual std::vector< std::vector< double > > fitTransform(
    const std::vector< std::vector< double > > & data
) =0
```

Fits the scaler to the data and then transforms the data. 

**Parameters**: 

  * **data** The input data as a vector of vectors of doubles. 


**Return**: std::vector<std::vector<double>> The normalized data. 

**Reimplemented by**: [nn::MinMaxScaler::fitTransform](classnn_1_1_min_max_scaler.md#function-fittransform), [nn::StandardScaler::fitTransform](classnn_1_1_standard_scaler.md#function-fittransform)
