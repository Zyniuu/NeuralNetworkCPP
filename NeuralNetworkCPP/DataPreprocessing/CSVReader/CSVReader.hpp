/**
 * C++ neural network library
 *
 * CSVReader.hpp
 */

#ifndef CSVREADER_HPP
#define CSVREADER_HPP

#include <string>
#include <vector>

namespace nn
{
    /**
     * @class CSVReader
     * @brief Reads data from a CSV file and separates it into features and labels.
     *
     * This class allows the user to specify the separator, whether labels are at the
     * beginning or end of each line, and whether the file has a header.
     */
    class CSVReader
    {
    private:
        std::string m_filename;                    ///< Path to the CSV file.
        char m_separator;                          ///< Character used to separate values in the CSV file.
        bool m_labelsAtEnd;                        ///< If true, labels are at the end of each line; otherwise, at the beginning.
        bool m_hasHeader;                          ///< If true, the first line of the file is treated as a header and skipped.
        std::vector<std::vector<double>> m_data;   ///< Stores the feature data from the CSV file.
        std::vector<std::vector<double>> m_labels; ///< Stores the labels from the CSV file.

    public:
        /**
         * @brief Constructs a CSVReader object.
         *
         * @param filename Path to the CSV file.
         * @param separator The character used to separate values in the CSV file (default: ',').
         * @param labelsAtEnd If true, labels are at the end of each line (default: true).
         * @param hasHeader If true, the first line of the file is treated as a header and skipped (default: false).
         */
        CSVReader(const std::string &filename, const char separator = ',', const bool labelsAtEnd = true, const bool hasHeader = false);

        /**
         * @brief Reads the CSV file and stores the data and labels.
         *
         * @throws std::runtime_error If the file cannot be opened or contains invalid data.
         */
        void read();

        /**
         * @brief Returns the feature data from the CSV file.
         *
         * @return std::vector<std::vector<double>> The feature data.
         */
        std::vector<std::vector<double>> getData() const { return m_data; };

        /**
         * @brief Returns the labels from the CSV file.
         *
         * @return std::vector<std::vector<double>> The labels.
         */
        std::vector<std::vector<double>> getLabels() const { return m_labels; };

    private:
        /**
         * @brief Splits a string into tokens based on the separator.
         *
         * @param line The input string.
         * @return std::vector<std::string> The tokens.
         */
        std::vector<std::string> split(const std::string &line) const;

        /**
         * @brief Converts a vector of strings to a vector of doubles.
         *
         * @param tokens The input tokens.
         * @return std::vector<double> The converted values.
         * @throws std::runtime_error If a token cannot be converted to a double.
         */
        std::vector<double> toDouble(const std::vector<std::string> &tokens) const;
    };
}

#endif