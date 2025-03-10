/**
 * C++ neural network library
 *
 * TestUtils.cpp
 */

#include <gtest/gtest.h>
#include <NeuralNetworkCPP/Utils/Utils.hpp>

TEST(UtilsTests, SliceValidRange)
{
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0},
        {7.0, 8.0}
    };

    std::vector<std::vector<double>> result = nn::slice(data, 1, 3);

    std::vector<std::vector<double>> expected = {
        {3.0, 4.0},
        {5.0, 6.0}
    };

    ASSERT_EQ(result, expected);
}

TEST(UtilsTests, SliceInvalidRange)
{
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    EXPECT_THROW(nn::slice(data, -1, 1), std::out_of_range);
    EXPECT_THROW(nn::slice(data, 0, 3), std::out_of_range);
    EXPECT_THROW(nn::slice(data, 2, 1), std::out_of_range);
}

TEST(UtilsTests, ReorderValid)
{
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };

    std::vector<int> order = {2, 0, 1};

    nn::reorderRows(data, order);

    std::vector<std::vector<double>> expected = {
        {5.0, 6.0},
        {1.0, 2.0},
        {3.0, 4.0}
    };

    ASSERT_EQ(data, expected);
}

TEST(UtilsTests, ReorderInvalid)
{
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    std::vector<int> invalidOrder1 = {2, 0};    // Index 2 is out of bounds
    std::vector<int> invalidOrder2 = {0, 1, 2}; // Size mismatch

    EXPECT_THROW(nn::reorderRows(data, invalidOrder1), std::out_of_range);
    EXPECT_THROW(nn::reorderRows(data, invalidOrder2), std::out_of_range);
}

TEST(UtilsTests, ShuffleValid)
{
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0},
        {7.0, 8.0},
        {9.0, 10.0},
        {11.0, 12.0},
        {13.0, 14.0},
        {15.0, 16.0},
        {17.0, 18.0}
    };

    std::vector<std::vector<double>> labels = {
        {1.0},
        {2.0},
        {3.0},
        {4.0},
        {5.0},
        {6.0},
        {7.0},
        {8.0},
        {9.0}
    };

    std::vector<std::vector<double>> dataShuffled = data;
    std::vector<std::vector<double>> labelsShuffled = labels;

    nn::shuffleDataset(dataShuffled, labelsShuffled);

    // Might not always pass
    EXPECT_NE(data, dataShuffled);
    EXPECT_NE(labels, labelsShuffled);
}

TEST(UtilsTests, ShuffleInvalid)
{
    std::vector<std::vector<double>> data = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    std::vector<std::vector<double>> labels = {
        {1.0}
    };

    std::vector<std::vector<double>> dataShuffled = data;
    std::vector<std::vector<double>> labelsShuffled = labels;

    EXPECT_THROW(nn::shuffleDataset(dataShuffled, labelsShuffled), std::runtime_error);
}

TEST(UtilsTests, ToCategorical)
{
    std::vector<std::vector<double>> labels = {
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    };

    std::vector<std::vector<double>> expectedOneHotEncoding = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };

    std::vector<std::vector<double>> output = nn::to_categorical(labels);

    EXPECT_EQ(expectedOneHotEncoding, output);
}

TEST(UtilsTests, ToCategoricalWIthNumOfClasses)
{
    std::vector<std::vector<double>> labels = {
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    };

    std::vector<std::vector<double>> expectedOneHotEncoding = {
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0}
    };

    std::vector<std::vector<double>> output = nn::to_categorical(labels, 4);

    EXPECT_EQ(expectedOneHotEncoding, output);
}

TEST(UtilsTests, ToCategoricalWIthWrongNumOfClasses)
{
    std::vector<std::vector<double>> labels = {
        {0.0},
        {1.0},
        {2.0},
        {3.0}
    };

    EXPECT_THROW(nn::to_categorical(labels, 3), std::runtime_error);
}