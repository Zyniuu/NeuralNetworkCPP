/**
 * C++ neural network library
 *
 * TestPreprocessing.cpp
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <NeuralNetworkCPP/DataPreprocessing/CSVReader/CSVReader.hpp>

TEST(CSVReaderTests, ReadCSVWithLabelsAtEnd)
{
        // Create a temporary CSV file for testing
        std::string filename = "test_csv_labels_at_end.csv";
        std::ofstream file(filename);
        file << "1.0,2.0,0\n";
        file << "3.0,4.0,1\n";
        file << "5.0,6.0,0\n";
        file.close();

        // Create a CSVReader object
        nn::CSVReader reader(filename, ',', true, false);

        // Read the CSV file
        reader.read();

        // Check the data
        std::vector<std::vector<double>> expectedData = {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0}
        };
        EXPECT_EQ(reader.getData(), expectedData);

        // Check the labels
        std::vector<std::vector<double>> expectedLabels = {
            {0.0},
            {1.0},
            {0.0}
        };
        EXPECT_EQ(reader.getLabels(), expectedLabels);

        // Clean up the temporary file
        std::filesystem::remove(filename);
}

TEST(CSVReaderTests, ReadCSVWithLabelsAtBeginning)
{
    // Create a temporary CSV file for testing
    std::string filename = "test_csv_labels_at_beginning.csv";
    std::ofstream file(filename);
    file << "0,1.0,2.0\n";
    file << "1,3.0,4.0\n";
    file << "0,5.0,6.0\n";
    file.close();

    // Create a CSVReader object
    nn::CSVReader reader(filename, ',', false, false);

    // Read the CSV file
    reader.read();

    // Check the data
    std::vector<std::vector<double>> expectedData = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };
    EXPECT_EQ(reader.getData(), expectedData);

    // Check the labels
    std::vector<std::vector<double>> expectedLabels = {
        {0.0},
        {1.0},
        {0.0}
    };
    EXPECT_EQ(reader.getLabels(), expectedLabels);

    // Clean up the temporary file
    std::filesystem::remove(filename);
}

TEST(CSVReaderTests, ReadCSVWithHeader)
{
    // Create a temporary CSV file for testing
    std::string filename = "test_csv_with_header.csv";
    std::ofstream file(filename);
    file << "feature1,feature2,label\n";
    file << "1.0,2.0,0\n";
    file << "3.0,4.0,1\n";
    file << "5.0,6.0,0\n";
    file.close();

    // Create a CSVReader object
    nn::CSVReader reader(filename, ',', true, true);

    // Read the CSV file
    reader.read();

    // Check the data
    std::vector<std::vector<double>> expectedData = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };
    EXPECT_EQ(reader.getData(), expectedData);

    // Check the labels
    std::vector<std::vector<double>> expectedLabels = {
        {0.0},
        {1.0},
        {0.0}
    };
    EXPECT_EQ(reader.getLabels(), expectedLabels);

    // Clean up the temporary file
    std::filesystem::remove(filename);
}

TEST(CSVReaderTests, ReadCSVWithInvalidToken)
{
    // Create a temporary CSV file for testing
    std::string filename = "test_csv_invalid_token.csv";
    std::ofstream file(filename);
    file << "1.0,2.0,0\n";
    file << "3.0,invalid,1\n";
    file << "5.0,6.0,0\n";
    file.close();

    // Create a CSVReader object
    nn::CSVReader reader(filename, ',', true, false);

    // Attempt to read the CSV file (should throw an exception)
    EXPECT_THROW(reader.read(), std::runtime_error);

    // Clean up the temporary file
    std::filesystem::remove(filename);
}