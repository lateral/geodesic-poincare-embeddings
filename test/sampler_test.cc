#include <random>
#include "gtest/gtest.h"
#include "sampler.h"

namespace {

TEST(SamplerTest, TestTrivial) {
    std::minstd_rand rng(1);
    std::vector<int64_t> counts = {1};
    std::vector<int32_t> exclude = {};
    poincare::Sampler sampler(1.0, counts, 100);
    for (int i = 0; i < 100; i++) {
        int32_t sample = sampler.get_sample(exclude, rng);
        EXPECT_EQ(sample, 0);
    }
}

TEST(SamplerTest, TestPowerOne) {
    std::minstd_rand rng(0);
    std::vector<int64_t> counts = {1, 2};
    std::vector<int32_t> exclude = {};
    int32_t sample_count = 50000;
    poincare::Sampler sampler(1.0, counts, sample_count);
    int32_t sum = 0;
    for (int i = 0; i < sample_count; i++) {
        sum += sampler.get_sample(exclude, rng);
    }
    EXPECT_NEAR(sum * 1. / sample_count, 0.67, 1e-2); 
}

TEST(SamplerTest, TestPowerZeroIsUniform) {
    std::minstd_rand rng(0);
    std::vector<int64_t> counts = {1, 2};
    std::vector<int32_t> exclude = {};
    int32_t sample_count = 50000;
    poincare::Sampler sampler(0.0, counts, sample_count);
    int32_t sum = 0;
    for (int i = 0; i < sample_count; i++) {
        sum += sampler.get_sample(exclude, rng);
    }
    EXPECT_NEAR(sum * 1. / sample_count, 0.5, 1e-2); 
}

TEST(SamplerTest, TestFractionalPower) {
    std::minstd_rand rng(0);
    std::vector<int64_t> counts = {1, 2};
    std::vector<int32_t> exclude = {};
    int32_t sample_count = 50000;
    poincare::Sampler sampler(0.75, counts, sample_count);
    int32_t sum = 0;
    for (int i = 0; i < sample_count; i++) {
        sum += sampler.get_sample(exclude, rng);
    }
    EXPECT_NEAR(sum * 1. / sample_count, pow(2, 0.75) / (1 + pow(2, 0.75)), 1e-2); 
}

TEST(SamplerTest, TestProbaZero) {
    std::minstd_rand rng(0);
    std::vector<int64_t> counts = {0, 1};
    std::vector<int32_t> exclude = {};
    int32_t sample_count = 500;
    poincare::Sampler sampler(1, counts, sample_count);
    int32_t sum = 0;
    for (int i = 0; i < sample_count; i++) {
        sum += sampler.get_sample(exclude, rng);
    }
    EXPECT_EQ(sum, sample_count); 
}

TEST(SamplerTest, TestExclude) {
    std::minstd_rand rng(0);
    std::vector<int64_t> counts = {3, 2, 3};
    std::vector<int32_t> exclude = {1};
    int32_t sample_count = 5000;
    poincare::Sampler sampler(1.0, counts, sample_count);
    for (int i = 0; i < sample_count; i++) {
        EXPECT_NE(sampler.get_sample(exclude, rng), 1);
    }
}

TEST(SamplerTest, TestSeedMakesDifference) {
    std::minstd_rand rng0(2);
    std::minstd_rand rng1(1);
    std::vector<int64_t> counts = {1, 1};
    std::vector<int32_t> exclude = {};
    int32_t sample_count = 10000;
    poincare::Sampler sampler(1.0, counts, sample_count);
    int32_t coincidence_count = 0;
    for (int i = 0; i < sample_count; i++) {
        coincidence_count += (sampler.get_sample(exclude, rng0) == sampler.get_sample(exclude, rng1));
    }
    EXPECT_LT(coincidence_count, sample_count);
}

}    // namespace
