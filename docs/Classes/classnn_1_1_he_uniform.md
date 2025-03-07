# nn::HeUniform



Implements He Uniform initialization for neural network weights.  [More...](#detailed-description)


`#include <HeUniform.hpp>`

Inherits from [nn::Initializer](classnn_1_1_initializer.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[HeUniform](classnn_1_1_he_uniform.md#function-heuniform)**(const int inputs, const int outputs)<br>Constructor for [HeUniform](classnn_1_1_he_uniform.md) initializer.  |
| virtual double | **[getRandomNum](classnn_1_1_he_uniform.md#function-getrandomnum)**() override<br>Generates a random number following He uniform distribution.  |

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
class nn::HeUniform;
```

Implements He Uniform initialization for neural network weights. 

He Uniform initialization is designed for networks using [ReLU](classnn_1_1_re_l_u.md) (or variants) activation functions. It draws weights from a uniform distribution within the range [-limit, limit], where limit = sqrt(6 / fan-in), and fan-in is the number of input neurons. 

## Public Functions Documentation

### function HeUniform

```cpp
HeUniform(
    const int inputs,
    const int outputs
)
```

Constructor for [HeUniform](classnn_1_1_he_uniform.md) initializer. 

**Parameters**: 

  * **inputs** Number of input neurons (fan-in). 
  * **outputs** Number of output neurons (fan-out). Not used in He Uniform initialization. 


Initializes the uniform distribution within the range [-limit, limit], where limit = sqrt(6 / fan-in).


### function getRandomNum

```cpp
virtual double getRandomNum() override
```

Generates a random number following He uniform distribution. 

**Return**: A randomly initialized value drawn from the uniform distribution. 

**Reimplements**: [nn::Initializer::getRandomNum](classnn_1_1_initializer.md#function-getrandomnum)
