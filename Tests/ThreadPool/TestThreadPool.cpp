/**
 * C++ neural network library
 *
 * TestThreadPool.cpp
 */

#include <gtest/gtest.h>
#include <NeuralNetworkCPP/GlobalThreadPool/GlobalThreadPool.hpp>

// Test if threading is working correctly
TEST(ThreadPoolTests, ParallelSum)
{
    constexpr int N = 1000;
    std::atomic<int> sum{0};
    auto &pool = nn::getGlobalThreadPool();

    std::vector<std::future<void>> futures;
    for (int i = 0; i < N; i++)
    {
        futures.emplace_back(pool.enqueue([&sum, i] {
            sum.fetch_add(i, std::memory_order_relaxed);
        }));
    }

    for (auto &f : futures)
        f.wait();

    int expected = (N * (N - 1)) / 2;
    EXPECT_EQ(sum.load(), expected);
}

// Test if function is executed correctly for small ranges
TEST(ThreadPoolTests, SmallRange)
{
    std::atomic<int> counter{0};
    auto &pool = nn::getGlobalThreadPool();

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 5; i++)
    {
        futures.emplace_back(pool.enqueue([&counter] {
            counter.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    for (auto &f : futures)
        f.wait();

    EXPECT_EQ(counter.load(), 5);
}

// Test if empty ranges are ignored
TEST(ThreadPoolTests, EdgeCaseEmptyRange)
{
    std::atomic<int> counter{0};
    auto &pool = nn::getGlobalThreadPool();

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 0; i++)
    {
        futures.emplace_back(pool.enqueue([&counter] {
            counter.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    for (auto &f : futures)
        f.wait();

    EXPECT_EQ(counter.load(), 0);
}

// Test if threads are reused
TEST(ThreadPoolTests, ThreadReuse)
{
    std::atomic<int> counter{0};
    auto &pool = nn::getGlobalThreadPool();

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 100; ++i)
    {
        futures.emplace_back(pool.enqueue([&counter] {
            counter.fetch_add(1, std::memory_order_relaxed);
        }));
    }

    for (auto &f : futures)
        f.wait();

    EXPECT_EQ(counter.load(), 100);
}

// Test if the parallelFor works correctly by counting the sum
TEST(ThreadPoolTests, ParallelForSum)
{
    constexpr int N = 1000;
    std::atomic<int> sum{0};
    auto &pool = nn::getGlobalThreadPool();

    pool.parallelFor(0, N, [&sum](int i) {
        sum.fetch_add(i, std::memory_order_relaxed);
    });

    int expected = (N * (N - 1)) / 2;
    EXPECT_EQ(sum.load(), expected);
}

// Test if the parallelFor ignores empty ranges
TEST(ThreadPoolTests, ParallelForEmptyRange)
{
    std::atomic<int> counter{0};
    auto &pool = nn::getGlobalThreadPool();

    pool.parallelFor(5, 5, [&counter](int) {
        counter.fetch_add(1, std::memory_order_relaxed);
    });

    EXPECT_EQ(counter.load(), 0);
}

// Test if the parallelFor works correctly for small ranges
TEST(ThreadPoolTests, ParallelForSmallRange)
{
    constexpr int N = 5;
    std::atomic<int> counter{0};
    auto &pool = nn::getGlobalThreadPool();

    pool.parallelFor(0, N, [&counter](int) {
        counter.fetch_add(1, std::memory_order_relaxed);
    });

    EXPECT_EQ(counter.load(), N);
}

// Test if the parallelFor modifies data correctly
TEST(ThreadPoolTests, ParallelForModifyArray)
{
    constexpr int N = 100;
    std::vector<int> data(N, 0);
    auto &pool = nn::getGlobalThreadPool();

    pool.parallelFor(0, N, [&data](int i) {
        data[i] = i * 2;
    });

    for (int i = 0; i < N; i++)
    {
        EXPECT_EQ(data[i], i * 2);
    }
}

// Test if the thread pool shuts down gracefully
TEST(ThreadPoolTests, Shutdown)
{
    auto &pool = nn::getGlobalThreadPool();

    // Submit a task
    auto future = pool.enqueue([] {
        return 42;
    });

    EXPECT_EQ(future.get(), 42); // Verify the task completes

    // Shutdown the pool
    pool.~ThreadPool();

    // Attempt to submit a new task after shutdown
    EXPECT_THROW({
        pool.enqueue([] {});
    }, std::runtime_error);
}