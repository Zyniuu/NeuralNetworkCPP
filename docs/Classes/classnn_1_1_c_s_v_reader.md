# nn::CSVReader



Reads data from a CSV file and separates it into features and labels.  [More...](#detailed-description)


`#include <CSVReader.hpp>`

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[CSVReader](classnn_1_1_c_s_v_reader.md#function-csvreader)**(const std::string & filename, const char separator =',', const bool labelsAtEnd =true, const bool hasHeader =false)<br>Constructs a [CSVReader](classnn_1_1_c_s_v_reader.md) object.  |
| void | **[read](classnn_1_1_c_s_v_reader.md#function-read)**()<br>Reads the CSV file and stores the data and labels.  |
| std::vector< std::vector< double > > | **[getData](classnn_1_1_c_s_v_reader.md#function-getdata)**() const<br>Returns the feature data from the CSV file.  |
| std::vector< std::vector< double > > | **[getLabels](classnn_1_1_c_s_v_reader.md#function-getlabels)**() const<br>Returns the labels from the CSV file.  |

## Detailed Description

```cpp
class nn::CSVReader;
```

Reads data from a CSV file and separates it into features and labels. 

This class allows the user to specify the separator, whether labels are at the beginning or end of each line, and whether the file has a header. 

## Public Functions Documentation

### function CSVReader

```cpp
CSVReader(
    const std::string & filename,
    const char separator =',',
    const bool labelsAtEnd =true,
    const bool hasHeader =false
)
```

Constructs a [CSVReader](classnn_1_1_c_s_v_reader.md) object. 

**Parameters**: 

  * **filename** Path to the CSV file. 
  * **separator** The character used to separate values in the CSV file (default: ','). 
  * **labelsAtEnd** If true, labels are at the end of each line (default: true). 
  * **hasHeader** If true, the first line of the file is treated as a header and skipped (default: false). 


### function read

```cpp
void read()
```

Reads the CSV file and stores the data and labels. 

**Exceptions**: 

  * **std::runtime_error** If the file cannot be opened or contains invalid data. 


### function getData

```cpp
inline std::vector< std::vector< double > > getData() const
```

Returns the feature data from the CSV file. 

**Return**: std::vector<std::vector<double>> The feature data. 

### function getLabels

```cpp
inline std::vector< std::vector< double > > getLabels() const
```

Returns the labels from the CSV file. 

**Return**: std::vector<std::vector<double>> The labels. 
