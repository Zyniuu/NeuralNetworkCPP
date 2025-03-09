# nn::Initializer



Abstract base class for weight initializers in neural networks.  [More...](#detailed-description)


`#include <Initializer.hpp>`

Inherited by [nn::HeNormal](classnn_1_1_he_normal.md), [nn::HeUniform](classnn_1_1_he_uniform.md), [nn::XavierNormal](classnn_1_1_xavier_normal.md), [nn::XavierUniform](classnn_1_1_xavier_uniform.md)

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[Initializer](classnn_1_1_initializer.md#function-initializer)**(const int inputs, const int outputs)<br>Constructor for the [Initializer](classnn_1_1_initializer.md) class.  |
| virtual double | **[getRandomNum](classnn_1_1_initializer.md#function-getrandomnum)**() =0<br>Pure virtual function for generating a random number.  |

## Protected Attributes

|                | Name           |
| -------------- | -------------- |
| int | **[m_inputs](classnn_1_1_initializer.md#variable-m_inputs)** <br>Number of input neurons.  |
| int | **[m_outputs](classnn_1_1_initializer.md#variable-m_outputs)** <br>Number of output neurons.  |
| std::mt19937 | **[m_gen](classnn_1_1_initializer.md#variable-m_gen)** <br>Mersenne Twister random number generator.  |

## Detailed Description

```cpp
class nn::Initializer;
```

Abstract base class for weight initializers in neural networks. 

This class provides a common interface for weight initialization strategies. Derived classes implement specific initialization methods (e.g., He Normal, Xavier Uniform). 

## Public Functions Documentation

### function Initializer

```cpp
inline Initializer(
    const int inputs,
    const int outputs
)
```

Constructor for the [Initializer](classnn_1_1_initializer.md) class. 

**Parameters**: 

  * **inputs** Number of input neurons. 
  * **outputs** Number of output neurons. 


Initializes the random number generator and stores the number of input and output neurons.


### function getRandomNum

```cpp
virtual double getRandomNum() =0
```

Pure virtual function for generating a random number. 

**Return**: A randomly initialized value. 

**Reimplemented by**: [nn::HeNormal::getRandomNum](classnn_1_1_he_normal.md#function-getrandomnum), [nn::HeUniform::getRandomNum](classnn_1_1_he_uniform.md#function-getrandomnum), [nn::XavierNormal::getRandomNum](classnn_1_1_xavier_normal.md#function-getrandomnum), [nn::XavierUniform::getRandomNum](classnn_1_1_xavier_uniform.md#function-getrandomnum)


Derived classes must implement this method to provide specific initialization logic.


## Protected Attributes Documentation

### variable m_inputs

```cpp
int m_inputs;
```

Number of input neurons. 

### variable m_outputs

```cpp
int m_outputs;
```

Number of output neurons. 

### variable m_gen

```cpp
std::mt19937 m_gen;
```

Mersenne Twister random number generator. 
