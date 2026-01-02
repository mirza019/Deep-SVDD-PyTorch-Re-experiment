# Deep SVDD Re-experiment: Comparison of Results

This document summarizes the results of the Deep SVDD re-experiment and compares them to the original paper's results.

## MNIST (Normal Class: 0)

| Experiment                        | AUC     |
| --------------------------------- | ------- |
| Original Paper                    | 98.0%   |
| User's Re-experiment              | 95.28%  |
| **Improved Experiment**           | **99.12%**  |

The improved experiment for MNIST outperforms both the user's re-experiment and the original paper. This was achieved by tuning the learning rate and the number of epochs to match the values reported in the original paper.

## CIFAR-10 (Normal Class: 0 - airplane)

| Experiment                        | AUC     |
| --------------------------------- | ------- |
| Original Paper                    | 61.7%   |
| User's Re-experiment              | 58.53%  |
| Improved Experiment (LeNet)       | 58.95%  |
| **Improved Experiment (ResNet-18)** | **43.61%**  |

### Analysis of CIFAR-10 Results

The experiments on CIFAR-10 were less successful.

*   The initial attempt to use a **ResNet-18** architecture with a low learning rate resulted in an AUC of **56.68%**.
*   The second attempt with a **LeNet** architecture, data augmentation, and tuned hyperparameters showed a slight improvement over the user's re-experiment, with an AUC of **58.95%**, but still did not outperform the original paper.
*   The final attempt with **ResNet-18**, more data augmentation, and a higher learning rate resulted in a very poor AUC of **43.61%**.

These results suggest that for this specific one-class classification task on CIFAR-10, the LeNet architecture with carefully tuned hyperparameters and data augmentation is more effective than the more complex ResNet-18 architecture. It is possible that the ResNet-18 architecture is too complex for this task and is prone to overfitting, or that it requires a more sophisticated pre-training strategy.

Further improvements for CIFAR-10 could be achieved by:
*   A more thorough hyperparameter search for the LeNet architecture.
*   Exploring different data augmentation techniques.
*   Investigating different pre-training strategies for the network.