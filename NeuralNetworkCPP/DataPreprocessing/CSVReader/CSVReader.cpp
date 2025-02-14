/**
 * C++ neural network library
 *
 * CSVReader.cpp
 */

#include "CSVReader.hpp"
#include <fstream>
#include <sstream>

namespace nn
{
    CSVReader::CSVReader(const std::string &filename, const char separator, const bool labelsAtEnd, const bool hasHeader)
        : m_filename(filename), m_separator(separator), m_labelsAtEnd(labelsAtEnd), m_hasHeader(hasHeader) {}

    void CSVReader::read()
    {
        // Open the file
        std::ifstream file(m_filename);
        if (!file.is_open())
            throw std::runtime_error("Failed to open file: " + m_filename);

        std::string line;

        // Skip the header line if specified
        if (m_hasHeader)
            std::getline(file, line);

        // Read file line by line
        while (std::getline(file, line))
        {
            // Split the line into tokens
            std::vector<std::string> tokens = split(line);
            if (tokens.empty())
                continue; // Skip empty lines

            // Convert tokens to doubles
            std::vector<double> values = toDouble(tokens);

            // Split the values into data and labels
            if (m_labelsAtEnd)
            {
                // Labels are at the end of the line
                std::vector<double> dataRow(values.begin(), values.end() - 1);
                std::vector<double> labelRow(values.end() - 1, values.end());
                m_data.push_back(dataRow);
                m_labels.push_back(labelRow);
            }
            else
            {
                // Labels are at the beginning of the line
                std::vector<double> labelRow(values.begin(), values.begin() + 1);
                std::vector<double> dataRow(values.begin() + 1, values.end());
                m_data.push_back(dataRow);
                m_labels.push_back(labelRow);
            }
        }
    }

    std::vector<std::string> CSVReader::split(const std::string &line) const
    {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(line);

        // Split the line into tokens based on the separator
        while (std::getline(tokenStream, token, m_separator))
            tokens.push_back(token);

        return tokens;
    }

    std::vector<double> CSVReader::toDouble(const std::vector<std::string> &tokens) const
    {
        std::vector<double> values;

        // Convert each token to a double
        for (const std::string &token : tokens)
        {
            try
            {
                values.push_back(std::stod(token));
            }
            catch (const std::invalid_argument &e)
            {
                throw std::runtime_error("Invalid token in CSV file: " + token);
            }
        }

        return values;
    }
}