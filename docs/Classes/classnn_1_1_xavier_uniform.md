# nn::XavierUniform



Implements Xavier (Glorot) uniform initialization for neural network weights.  [More...](#detailed-description)


`#include <XavierUniform.hpp>`

Inherits from [nn::Initializer](classnn_1_1_initializer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[XavierUniform](classnn_1_1_xavier_uniform.md#function-xavieruniform)**(const int inputs, const int outputs)<br>Constructor for [XavierUniform](classnn_1_1_xavier_uniform.md) initializer.  |
| virtual double | **[getRandomNum](classnn_1_1_xavier_uniform.md#function-getrandomnum)**() override<br>Generates a random number following Xavier uniform distribution.  |

## Additional inherited members

**Public Functions inherited from [nn::Initializer](classnn_1_1_initializer.md)**

|                | Name           |
| -------------- | -------------- |
| | **[Initializer](classnn_1_1_initializer.md#function-initializer)**(const int inputs, const int outputs)<br>Constructor for the [Initializer](classnn_1_1_initializer.md) class.  |

**Protected Attributes inherited from [nn::Initializer](classnn_1_1_initializer.md)**

|                | Name           |
| -------------- | -------------- |
| int | **[m_inputs](classnn_1_1_initializer.md#variable-m_inputs)** <br>Number of input neurons.  |
| int | **[m_outputs](classnn_1_1_initializer.md#variable-m_outputs)** <br>Number of output neurons.  |
| std::mt19937 | **[m_gen](classnn_1_1_initializer.md#variable-m_gen)** <br>Mersenne Twister random number generator.  |


## Detailed Description

```cpp
class nn::XavierUniform;
```

Implements Xavier (Glorot) uniform initialization for neural network weights. 

Xavier Uniform initialization is designed for networks using sigmoid or tanh activation functions. It draws weights from a uniform distribution within the range [-limit, limit], where limit = sqrt(6 / (fan-in + fan-out)), and fan-in and fan-out are the number of input and output neurons. 

## Public Functions Documentation

### function XavierUniform

```cpp
XavierUniform(
    const int inputs,
    const int outputs
)
```

Constructor for [XavierUniform](classnn_1_1_xavier_uniform.md) initializer. 

**Parameters**: 

  * **inputs** Number of input neurons (fan-in). 
  * **outputs** Number of output neurons (fan-out). 


Initializes the uniform distribution within the range [-limit, limit], where limit = sqrt(6 / (fan-in + fan-out)).


### function getRandomNum

```cpp
virtual double getRandomNum() override
```

Generates a random number following Xavier uniform distribution. 

**Return**: A randomly initialized value drawn from the uniform distribution. 

**Reimplements**: [nn::Initializer::getRandomNum](classnn_1_1_initializer.md#function-getrandomnum)
