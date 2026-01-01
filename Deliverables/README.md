# Deliverables

This folder contains the deliverables for the "Hybrid Explainable One-Class Framework" project, including generated plots and a summary of experimental results.

## 1. Individual Experiment Plots

The following plots are generated *per experiment* and are saved in their respective subdirectories within `Deliverables/Plots/`:

*   **Normal and Outlier Samples:** Images of the most normal and most anomalous samples detected by the model.
*   **t-SNE Latent-Space Visualization:** Visualizations of the latent space for individual experiments.
*   **Grad-CAM Overlays:** Grad-CAM heatmaps for selected normal and anomalous samples from individual experiments.

**Note:** All generated `.png` plot files from the experiment log directories have been moved to their respective subdirectories within `Deliverables/Plots/` for organized access.

## 2. Comparative Plots

These plots provide a comparison of the performance and interpretability across different models (baseline, best hybrid, adaptive threshold) for each dataset. They are located in `Deliverables/Plots/Comparative/`.

*   **ROC Curves:**
    *   `Fashion-MNIST_ROC_Curve.png`
    *   `MNIST_ROC_Curve.png`
    *   `CIFAR-10_ROC_Curve.png`
*   **t-SNE Latent-Space Visualization:**
    *   `Fashion-MNIST_tSNE_Latent_Space.png`
    *   `MNIST_tSNE_Latent_Space.png`
    *   `CIFAR-10_tSNE_Latent_Space.png`
*   **Grad-CAM Overlays:**
    *   `Fashion-MNIST_Grad_CAM_Comparative.png`
    *   `MNIST_Grad_CAM_Comparative.png`
    *   `CIFAR-10_Grad_CAM_Comparative.png`

## 3. Tables

*   **Comparative AUC and F1 Score Summary:**
    A `summary_table.md` has been generated in this `Deliverables` folder. It provides a comparative overview of AUC and F1 scores for all executed experiments across different datasets and configurations (baseline, hybrid with varying `mu1`/`mu2`, and adaptive thresholding). This table helps in assessing the performance improvements and robustness.

## 4. Datasets

The project utilizes the following datasets:

*   **MNIST**
*   **Fashion-MNIST**
*   **CIFAR-10 subset**

These datasets are automatically downloaded and preprocessed by the `srcv2/datasets` module when you run an experiment, provided they are not already present in your specified `data_path`.

## 5. Execution Environment

All experiments were executed on a **MacBook Air M3 CPU**. The code is designed to be lightweight and efficient for CPU execution. You can specify the device using the `--device cpu` flag when running `srcv2/main.py`.
