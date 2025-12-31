# Deliverables

This folder contains information and instructions on how to generate the deliverables for the "Hybrid Explainable One-Class Framework" project.

## 1. Plots

The following plots are generated as part of the experiment execution and are saved in the respective experiment's log directory (e.g., `log/mnist_test/`).

*   **ROC Curves:** The AUC-ROC score is calculated during the `test` phase of each experiment. While a direct ROC curve plot might not be explicitly saved as an image, the AUC score is a key metric derived from it. You can generate ROC curves from the `test_scores` saved in `results.json` if needed.
*   **t-SNE Latent-Space Visualization:** These visualizations are generated and saved as `tsne_latent.png` (or similar, e.g., `tsne_latent_cifar10.png`) within the experiment's log directory.
*   **Grad-CAM Overlays:** If the `--grad_cam` flag is set to `True` during experiment execution, Grad-CAM heatmaps are generated for selected normal and anomalous samples and saved as `normal_grad_cam.png` and `outlier_grad_cam.png` in the experiment's log directory.

**To generate these plots:**

Run your desired experiments using `srcv2/main.py`. For example:

```bash
python srcv2/main.py mnist mnist_LeNet log/mnist_test data --normal_class 0 --device cpu --grad_cam True
```

Replace `mnist_test` with your desired experiment name and adjust parameters as needed. The generated plots will be found in `log/your_experiment_name/`.

## 2. Tables

*   **Comparative AUC, Noise Robustness, and Ablation (λ1, λ2) Tables:**
    The results of each experiment, including AUC scores and other metrics, are saved in a `results.json` file within each experiment's log directory (e.g., `log/mnist_test/results.json`).

    To generate comparative tables, you would typically run multiple experiments with varying parameters (e.g., different `mu1`, `mu2` for ablation, different `noise_std` for noise robustness) and then write a separate Python script to parse these `results.json` files and aggregate the data into a table format.

    Here's a conceptual example of how you might structure a Python script to read and compare results:

    ```python
    import json
    import os

    def load_results(experiment_path):
        results_file = os.path.join(experiment_path, 'results.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
        return None

    def compare_experiments(experiment_dirs):
        all_results = {}
        for exp_dir in experiment_dirs:
            results = load_results(exp_dir)
            if results:
                exp_name = os.path.basename(exp_dir)
                all_results[exp_name] = {
                    'test_auc': results.get('test_auc'),
                    'test_f1': results.get('test_f1'),
                    # Add other metrics you want to compare
                }
        
        # You can then print or format these results into a table
        print("| Experiment Name | AUC     | F1 Score |")
        print("|-----------------|---------|----------|")
        for name, metrics in all_results.items():
            print(f"| {name:<15} | {metrics['test_auc']:.4f} | {metrics['test_f1']:.4f} |")

    if __name__ == "__main__":
        # Example usage:
        # Assuming your experiment logs are in 'log/exp1', 'log/exp2', etc.
        experiment_paths = [
            'log/mnist_test',
            'log/fashion_mnist_hybrid_test',
            'log/fashion_mnist_noise_0.1_test'
            # Add all relevant experiment paths here
        ]
        compare_experiments(experiment_paths)
    ```

## 3. Datasets

The project utilizes the following datasets:

*   **MNIST**
*   **Fashion-MNIST**
*   **CIFAR-10 subset**

These datasets are automatically downloaded and preprocessed by the `srcv2/datasets` module when you run an experiment, provided they are not already present in your specified `data_path`.

## 4. Execution Environment

All experiments are planned to be executed on a **MacBook Air M3 CPU**. The code is designed to be lightweight and efficient for CPU execution. You can specify the device using the `--device cpu` flag when running `srcv2/main.py`.
