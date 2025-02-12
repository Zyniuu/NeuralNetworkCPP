/**
 * C++ neural network library
 *
 * BatchNormalization.hpp
 */

#ifndef BATCHNORMALIZATION_HPP
#define BATCHNORMALIZATION_HPP

#include "../Common/Layer.hpp"

namespace nn
{
    class BatchNormalization : public Layer
    {
    private:
        Matrix m_gamma;
        Matrix m_beta;
        Matrix m_runningMean;
        Matrix m_runningVar;
        Matrix m_input;
        double m_epsilon;
        double m_momentum;

    public:
        BatchNormalization(const int numFeatures, const double epsilon = 1e-15, const double momentum = 0.9);

        Matrix forward(const Matrix &input) override;
        Matrix backward(const Matrix &gradient) override;
        void resetGradients() override;
        void applyGradient(Optimizer &optimizer, const int batchSize) override;
        void save(std::ofstream &file) const override;
        e_layerType getType() const override { return BATCH_NORM; };
    };
}

#endif