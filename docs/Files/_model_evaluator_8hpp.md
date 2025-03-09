# ModelParts/ModelEvaluator/ModelEvaluator.hpp



## Namespaces

| Name           |
| -------------- |
| **[nn](../Namespaces/namespacenn.md)**  |

## Classes

|                | Name           |
| -------------- | -------------- |
| class | **[nn::ModelEvaluator](../Classes/classnn_1_1_model_evaluator.md)** <br>Handles model evaluation, prediction, and metric computation.  |




## Source code

```cpp


#include "../ModelLayers/ModelLayers.hpp"
#include "../../Logger/Logger.hpp"

namespace nn
{
    class ModelEvaluator : public ModelLayers
    {
    public:
        std::vector<double> predict(const std::vector<double> &input);

        std::vector<std::vector<double>> predict(const std::vector<std::vector<double>> &input);

        double evaluate(
            const std::vector<std::vector<double>> &xTest,
            const std::vector<std::vector<double>> &yTest,
            const e_metric metric = ACCURACY_LOG
        );

    protected:
        Matrix forward(const Matrix &input);

        std::vector<double> evaluate(
            const std::vector<std::vector<double>> &xTest,
            const std::vector<std::vector<double>> &yTest,
            const std::vector<e_metric> &metrics
        );

    private:
        void setBatchTrainingMode(const bool isTraining);

        double computeMetric(
            const std::vector<std::vector<double>> &predictions,
            const std::vector<std::vector<double>> &targets,
            const e_metric metric
        );

        double computeAccuracy(
            const std::vector<std::vector<double>> &predictions,
            const std::vector<std::vector<double>> &targets
        );

        double computeMAE(
            const std::vector<std::vector<double>> &predictions,
            const std::vector<std::vector<double>> &targets
        );
    };
}
```
