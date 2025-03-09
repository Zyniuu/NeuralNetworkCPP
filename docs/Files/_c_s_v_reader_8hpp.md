# DataPreprocessing/CSVReader/CSVReader.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::CSVReader](../Classes/classnn_1_1_c_s_v_reader.md)** <br>Reads data from a CSV file and separates it into features and labels.  |




## Source code

```cpp


#ifndef CSVREADER_HPP
#define CSVREADER_HPP

#include <string>
#include <vector>

namespace nn
{
    class CSVReader
    {
    private:
        std::string m_filename;                    
        char m_separator;                          
        bool m_labelsAtEnd;                        
        bool m_hasHeader;                          
        std::vector<std::vector<double>> m_data;   
        std::vector<std::vector<double>> m_labels; 

    public:
        CSVReader(const std::string &filename, const char separator = ',', const bool labelsAtEnd = true, const bool hasHeader = false);

        void read();

        std::vector<std::vector<double>> getData() const { return m_data; };

        std::vector<std::vector<double>> getLabels() const { return m_labels; };

    private:
        std::vector<std::string> split(const std::string &line) const;

        std::vector<double> toDouble(const std::vector<std::string> &tokens) const;
    };
}

#endif
```
