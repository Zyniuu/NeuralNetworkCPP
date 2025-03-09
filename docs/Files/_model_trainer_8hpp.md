# ModelParts/ModelTrainer/ModelTrainer.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::ModelTrainer](../Classes/classnn_1_1_model_trainer.md)** <br>Handles the training of a neural network model.  |




## Source code

```cpp


#include "../ModelEvaluator/ModelEvaluator.hpp"
#include "../../Losses/Losses.hpp"
#include "../../Optimizers/Optimizers.hpp"

namespace nn
{
    class ModelTrainer : public ModelEvaluator
    {
    private:
        std::unique_ptr<Optimizer> m_optimizer; 
        std::unique_ptr<Loss> m_loss;           
        std::unique_ptr<Logger> m_logger;       
        std::vector<e_metric> m_metrics;        

    public:
        void backward(const Matrix &gradient);

        void compile(
            std::unique_ptr<Optimizer> optimizer,
            std::unique_ptr<Loss> lossFunc,
            const std::vector<e_metric> &metrics = {}
        );

        bool train(
            const std::vector<std::vector<double>> &xTrain,
            const std::vector<std::vector<double>> &yTrain,
            const int epochs,
            const int batchSize = 1,
            const double validationSplit = 0.0,
            const int patience = 10,
            const double minDelta = 0.0001,
            const bool verbose = true
        );

    private:
        void trainOnBatch(
            const std::vector<std::vector<double>> &xBatch,
            const std::vector<std::vector<double>> &yBatch,
            double &loss
        );
    };
}
```
