/**
 * C++ neural network library
 *
 * TestPreprocessing.cpp
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <NeuralNetworkCPP/DataPreprocessing/CSVReader/CSVReader.hpp>
#include <NeuralNetworkCPP/DataPreprocessing/Scalers/Scalers.hpp>

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

TEST(ScalersTests, StandardScalerFitTransform)
{
    // Create a StandardScaler object
    nn::StandardScaler scaler;

    // Input data
    std::vector<std::vector<double>> data = {
        {0.0, 0.0},
        {0.0, 0.0},
        {1.0, 1.0},
        {1.0, 1.0}
    };

    // Fit and transform the data
    auto normalizedData = scaler.fitTransform(data);

    // Expected normalized data
    std::vector<std::vector<double>> expectedData = {
        {-1.0, -1.0},
        {-1.0, -1.0},
        {1.0, 1.0},
        {1.0, 1.0}
    };

    // Check if the normalized data matches the expected data
    for (size_t i = 0; i < normalizedData.size(); ++i)
    {
        for (size_t j = 0; j < normalizedData[i].size(); ++j)
        {
            EXPECT_EQ(normalizedData[i][j], expectedData[i][j]);
        }
    }
}

TEST(ScalersTests, StandardScalerEmptyData)
{
    // Create a StandardScaler object
    nn::StandardScaler scaler;

    // Empty input data
    std::vector<std::vector<double>> data;

    // Attempt to fit and transform (should throw an exception)
    EXPECT_THROW(scaler.fitTransform(data), std::runtime_error);
}

TEST(ScalersTests, MinMaxScalerFitTransform)
{
    // Create a MinMaxScaler object
    nn::MinMaxScaler scaler(0.0, 1.0);

    // Input data
    std::vector<std::vector<double>> data = {
        {-1.0, 2.0},
        {-0.5, 6.0},
        {0.0, 10.0},
        {1.0, 18.0}
    };

    // Fit and transform the data
    auto normalizedData = scaler.fitTransform(data);

    // Expected normalized data
    std::vector<std::vector<double>> expectedData = {
        {0.0, 0.0},
        {0.25, 0.25},
        {0.5, 0.5},
        {1.0, 1.0}
    };

    // Check if the normalized data matches the expected data
    for (size_t i = 0; i < normalizedData.size(); ++i)
    {
        for (size_t j = 0; j < normalizedData[i].size(); ++j)
        {
            EXPECT_EQ(normalizedData[i][j], expectedData[i][j]);
        }
    }
}

TEST(ScalersTests, MinMaxScalerEmptyData)
{
    // Create a MinMaxScaler object
    nn::MinMaxScaler scaler;

    // Empty input data
    std::vector<std::vector<double>> data;

    // Attempt to fit and transform (should throw an exception)
    EXPECT_THROW(scaler.fitTransform(data), std::runtime_error);
}