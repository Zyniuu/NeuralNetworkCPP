# nn::XavierNormal



Implements Xavier (Glorot) normal initialization for neural network weights.  [More...](#detailed-description)


`#include <XavierNormal.hpp>`

Inherits from [nn::Initializer](classnn_1_1_initializer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[XavierNormal](classnn_1_1_xavier_normal.md#function-xaviernormal)**(const int inputs, const int outputs)<br>Constructor for [XavierNormal](classnn_1_1_xavier_normal.md) initializer.  |
| virtual double | **[getRandomNum](classnn_1_1_xavier_normal.md#function-getrandomnum)**() override<br>Generates a random number following Xavier normal distribution.  |

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
class nn::XavierNormal;
```

Implements Xavier (Glorot) normal initialization for neural network weights. 

Xavier Normal initialization is designed for networks using sigmoid or tanh activation functions. It draws weights from a normal distribution with a mean of 0 and a standard deviation of sqrt(2 / (fan-in + fan-out)), where fan-in is the number of input neurons and fan-out is the number of output neurons. 

## Public Functions Documentation

### function XavierNormal

```cpp
XavierNormal(
    const int inputs,
    const int outputs
)
```

Constructor for [XavierNormal](classnn_1_1_xavier_normal.md) initializer. 

**Parameters**: 

  * **inputs** Number of input neurons (fan-in). 
  * **outputs** Number of output neurons (fan-out). 


Initializes the normal distribution with a standard deviation of sqrt(2 / (fan-in + fan-out)).


### function getRandomNum

```cpp
virtual double getRandomNum() override
```

Generates a random number following Xavier normal distribution. 

**Return**: A randomly initialized value drawn from the normal distribution. 

**Reimplements**: [nn::Initializer::getRandomNum](classnn_1_1_initializer.md#function-getrandomnum)
