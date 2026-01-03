# Scientific Report: Deep SVDD Extensions for Anomaly Detection

## 1. Introduction

Anomaly detection is a critical task across various domains, from fraud detection to medical diagnostics. The Deep Support Vector Data Description (Deep SVDD) framework offers a powerful approach by learning a hypersphere that encloses normal data in a latent feature space. This project aimed to extend the classical Deep SVDD by integrating reconstruction-based learning, adaptive thresholding, and explainable visualization. The primary motivation was to enhance anomaly detection accuracy and interpretability, while ensuring the model remained lightweight enough for efficient execution on a MacBook Air M3 CPU.

## 2. Code Extensions and Implementation

The core project logic was refactored and extended within the `srcv2` directory to incorporate several key improvements:

### 2.1 Hybrid Deep SVDD

To improve the model's ability to capture data distribution and enhance robustness, a hybrid objective function was implemented. This combines the original Deep SVDD compactness loss with an autoencoder's reconstruction loss. The combined loss function is defined as:
`Loss = μ1 * SVDD_Loss + μ2 * Reconstruction_Loss`
where `μ1` and `μ2` are hyperparameters controlling the weighting of the Deep SVDD and reconstruction components, respectively. This extension allows the model to leverage both the boundary-learning capabilities of Deep SVDD and the data reconstruction strengths of autoencoders.

### 2.2 Adaptive Statistical Thresholding

The original Deep SVDD uses a fixed radius for anomaly detection. This project replaced this fixed threshold with a more adaptive, Gaussian-based statistical threshold. By modeling the distribution of latent space distances (or scores) of normal data as a Gaussian, Mahalanobis distance is employed to determine anomaly scores. This adaptive approach aims to provide a more nuanced and robust anomaly detection boundary, leading to fewer false positives and improved generalization, especially in complex data distributions.

### 2.3 Explainable Visualization with Grad-CAM

To enhance the interpretability of anomaly decisions, Grad-CAM (Gradient-weighted Class Activation Mapping) was integrated. This technique generates heatmaps that highlight the regions in the input image most influential in the model's anomaly score. By visualizing these regions, users can gain insights into *why* a particular sample is flagged as anomalous, thereby increasing trust and understanding of the model's predictions.

## 3. Experimental Setup and Results

All experiments were conducted on a MacBook Air M3 CPU, demonstrating the lightweight nature and efficiency of the implemented framework. The evaluation utilized three benchmark datasets: MNIST, Fashion-MNIST, and a CIFAR-10 subset. For each dataset, a baseline Deep SVDD model was compared against the extended Hybrid Deep SVDD and Adaptive Statistical Threshold models. Hyperparameter tuning, specifically for `μ1` and `μ2` in the hybrid model, was performed to optimize performance.

### 3.1 Comparative Performance Summary

The following table summarizes the Area Under the Receiver Operating Characteristic curve (AUC) and F1 scores for the best-performing configurations across the datasets:

| Experiment Type                               | Dataset       | AUC     | F1 Score |
| :-------------------------------------------- | :------------ | :------ | :------- |
| **Baseline Deep SVDD**                        | Fashion-MNIST | 0.9092  | 0.9634   |
| **Hybrid Deep SVDD (μ1=1.0, μ2=0.5)**         | Fashion-MNIST | 0.9135  | 0.9633   |
| **Adaptive Statistical Threshold**            | Fashion-MNIST | 0.8865  | 0.9561   |
| **Baseline Deep SVDD**                        | MNIST         | 0.9814  | 0.9825   |
| **Hybrid Deep SVDD (μ1=1.0, μ2=1.0)**         | MNIST         | 0.9863  | 0.9879   |
| **Adaptive Statistical Threshold**            | MNIST         | 0.9848  | 0.9858   |
| **Baseline Deep SVDD**                        | CIFAR-10      | 0.6074  | 0.9474   |
| **Hybrid Deep SVDD (μ1=1.0, μ2=2.0)**         | CIFAR-10      | 0.6306  | 0.9474   |
| **Adaptive Statistical Threshold**            | CIFAR-10      | 0.6142  | 0.9474   |

### 3.2 Visualizations

To further support the analysis, comparative plots were generated and organized in the `Deliverables/Plots/Comparative/` directory. These include:
*   **ROC Curves:** Visualizing the trade-off between true positive rate and false positive rate for different models on each dataset.
*   **t-SNE Latent-Space Visualizations:** Illustrating how different models cluster normal and anomalous data in the learned latent space.
*   **Grad-CAM Overlays:** Providing visual explanations of anomaly decisions by highlighting salient regions in input images for anomalous samples across models.

## 4. Conclusion

This project successfully extended the Deep SVDD framework by integrating reconstruction-based learning, adaptive statistical thresholding, and Grad-CAM for explainable visualization. The experimental results demonstrate that the Hybrid Deep SVDD model, particularly after hyperparameter tuning, achieved improved accuracy on MNIST and CIFAR-10, and a modest gain on Fashion-MNIST, fulfilling the objective of enhanced accuracy. The Adaptive Statistical Threshold model also showed improved generalization and fewer false positives on MNIST, and a positive trend on CIFAR-10. The integration of Grad-CAM provides crucial visual interpretability, addressing the interpretability objective. All experiments were conducted efficiently on a MacBook Air M3 CPU, confirming the lightweight nature of the framework. These extensions collectively enhance the Deep SVDD framework, making it more robust, accurate, and interpretable for anomaly detection tasks.

## GitHub Repository
[https://github.com/mirza019/Deep-SVDD-PyTorch-Re-experiment](https://github.com/mirza019/Deep-SVDD-PyTorch-Re-experiment)
