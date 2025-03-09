# nn::HeNormal



Implements He Normal initialization for neural network weights.  [More...](#detailed-description)


`#include <HeNormal.hpp>`

Inherits from [nn::Initializer](classnn_1_1_initializer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[HeNormal](classnn_1_1_he_normal.md#function-henormal)**(const int inputs, const int outputs)<br>Constructor for [HeNormal](classnn_1_1_he_normal.md) initializer.  |
| virtual double | **[getRandomNum](classnn_1_1_he_normal.md#function-getrandomnum)**() override<br>Generates a random number following He normal distribution.  |

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
class nn::HeNormal;
```

Implements He Normal initialization for neural network weights. 

He Normal initialization is designed for networks using [ReLU](classnn_1_1_re_l_u.md) (or variants) activation functions. It draws weights from a normal distribution with a mean of 0 and a standard deviation of sqrt(2 / fan-in), where fan-in is the number of input neurons. 

## Public Functions Documentation

### function HeNormal

```cpp
HeNormal(
    const int inputs,
    const int outputs
)
```

Constructor for [HeNormal](classnn_1_1_he_normal.md) initializer. 

**Parameters**: 

  * **inputs** Number of input neurons (fan-in). 
  * **outputs** Number of output neurons (fan-out). Not used in He Normal initialization. 


Initializes the normal distribution with a standard deviation of sqrt(2 / fan-in).


### function getRandomNum

```cpp
virtual double getRandomNum() override
```

Generates a random number following He normal distribution. 

**Return**: A randomly initialized value drawn from the normal distribution. 

**Reimplements**: [nn::Initializer::getRandomNum](classnn_1_1_initializer.md#function-getrandomnum)
