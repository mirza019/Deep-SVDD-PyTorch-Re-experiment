# Deliverables

This folder contains the deliverables for the "Hybrid Explainable One-Class Framework" project, including generated plots and a summary of experimental results.

## 1. Plots

The following plots are generated as part of the experiment execution:

*   **ROC Curves:** The AUC-ROC score is calculated during the `test` phase of each experiment. While a direct ROC curve plot might not be explicitly saved as an image, the AUC score is a key metric derived from it. You can generate ROC curves from the `test_scores` saved in `results.json` if needed.
*   **t-SNE Latent-Space Visualization:** These visualizations are generated and saved as `tsne_latent.png` (or similar, e.g., `tsne_latent_cifar10.png`).
*   **Grad-CAM Overlays:** If the `--grad_cam` flag is set to `True` during experiment execution, Grad-CAM heatmaps are generated for selected normal and anomalous samples and saved as `normal_grad_cam.png` and `outlier_grad_cam.png`.

**Note:** All generated `.png` plot files from the experiment log directories have been moved to their respective subdirectories within `Deliverables/Plots/` for organized access.

## 2. Tables

*   **Comparative AUC and F1 Score Summary:**
    A `summary_table.md` has been generated in this `Deliverables` folder. It provides a comparative overview of AUC and F1 scores for all executed experiments across different datasets and configurations (baseline, hybrid with varying `mu1`/`mu2`, and adaptive thresholding). This table helps in assessing the performance improvements and robustness.

## 3. Datasets

The project utilizes the following datasets:

*   **MNIST**
*   **Fashion-MNIST**
*   **CIFAR-10 subset**

These datasets are automatically downloaded and preprocessed by the `srcv2/datasets` module when you run an experiment, provided they are not already present in your specified `data_path`.

## 4. Execution Environment

All experiments were executed on a **MacBook Air M3 CPU**. The code is designed to be lightweight and efficient for CPU execution. You can specify the device using the `--device cpu` flag when running `srcv2/main.py`.