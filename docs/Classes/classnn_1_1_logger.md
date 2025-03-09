# nn::Logger



Handles logging of training progress and metrics. 


`#include <Logger.hpp>`

## Public Functions

|                | Name           |
| -------------- | -------------- |
| | **[Logger](classnn_1_1_logger.md#function-logger)**(const int progressBarLength =30)<br>Constructs a [Logger](classnn_1_1_logger.md) with the specified metrics.  |
| void | **[logTrainingStart](classnn_1_1_logger.md#function-logtrainingstart)**()<br>Starts the training timer.  |
| void | **[logTrainingEnd](classnn_1_1_logger.md#function-logtrainingend)**(const bool isEarlyStopped)<br>Stops the training timer and logs the end of training.  |
| void | **[logEpochStart](classnn_1_1_logger.md#function-logepochstart)**(const int currentEpoch, const int totalEpochs)<br>Logs the start of an epoch.  |
| void | **[logEpochEnd](classnn_1_1_logger.md#function-logepochend)**(const int totalBatches, const double loss, const std::vector< double > & computedMetrics, const std::vector< [e_metric](../Namespaces/namespacenn.md#enum-e_metric) > & metrics)<br>Logs the end of an epoch.  |
| void | **[logBatch](classnn_1_1_logger.md#function-logbatch)**(const int currentBatch, const int totalBatches)<br>Logs the progress of a batch.  |

## Public Functions Documentation

### function Logger

```cpp
Logger(
    const int progressBarLength =30
)
```

Constructs a [Logger](classnn_1_1_logger.md) with the specified metrics. 

**Parameters**: 

  * **progressBarLength** Max length of the progress bar. 


### function logTrainingStart

```cpp
void logTrainingStart()
```

Starts the training timer. 

### function logTrainingEnd

```cpp
void logTrainingEnd(
    const bool isEarlyStopped
)
```

Stops the training timer and logs the end of training. 

**Parameters**: 

  * **isEarlyStopped** Flag for displaying end of training or early stopped message. 


### function logEpochStart

```cpp
void logEpochStart(
    const int currentEpoch,
    const int totalEpochs
)
```

Logs the start of an epoch. 

**Parameters**: 

  * **currentEpoch** Current epoch number. 
  * **totalEpochs** Total number of epochs. 


### function logEpochEnd

```cpp
void logEpochEnd(
    const int totalBatches,
    const double loss,
    const std::vector< double > & computedMetrics,
    const std::vector< e_metric > & metrics
)
```

Logs the end of an epoch. 

**Parameters**: 

  * **totalBatches** Total number of baches. 
  * **loss** Computed loss between predictions and targets. 
  * **computedMetrics** Computed metrics on validation set. 
  * **metrics** Metrics to log. 


### function logBatch

```cpp
void logBatch(
    const int currentBatch,
    const int totalBatches
)
```

Logs the progress of a batch. 

**Parameters**: 

  * **currentBatch** Current batch number. 
  * **totalBatches** Total number of batches. 
