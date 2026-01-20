# Running Experiments: Baseline vs. Hybrid Models

This document provides clear instructions on how to run the baseline (original) Deep SVDD models and the new Hybrid Deep SVDD models, along with adaptive thresholding and Grad-CAM visualizations, to reproduce the results and explore the implemented extensions.

All experiments are designed to run on a CPU (e.g., MacBook Air M3 CPU).

## Prerequisites

Before running any experiments, ensure you have the following:

1.  **Python 3.x** installed.
2.  **Virtual Environment:** It is highly recommended to use a virtual environment. This project uses `myenv`.
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```
3.  **Dependencies:** Install the required Python packages.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The `requirements.txt` file should contain all necessary libraries like `torch`, `numpy`, `scikit-learn`, `matplotlib`, `opencv-python`, etc.)*

## Running Experiments with `srcv2` (Recommended)

All experiments are executed using the `srcv2/main.py` script. This script allows you to configure various aspects of the Deep SVDD model, including the dataset, network architecture, objective, hybrid loss parameters, thresholding method, and whether to generate Grad-CAM visualizations.

### Common Parameters:

*   `dataset_name`: `mnist`, `fashion_mnist`, `cifar10`
*   `net_name`: `mnist_LeNet`, `cifar10_LeNet`, `cifar10_LeNet_ELU` (choose based on dataset)
*   `xp_path`: Path to save experiment logs and results (e.g., `log/my_experiment_name`)
*   `data_path`: Root path where datasets will be downloaded/stored (e.g., `data`)
*   `--normal_class`: The class considered as normal (e.g., `0`)
*   `--device`: Computation device (`cpu` for MacBook Air M3 CPU)
*   `--pretrain`: Whether to pretrain the network via autoencoder (`True` or `False`)

### 1. Running Baseline (Original) Deep SVDD Models

To run the original Deep SVDD model without hybrid loss or adaptive thresholding, set `--hybrid False` and `--thresholding fixed`.

**Example: Baseline Deep SVDD on Fashion-MNIST (Normal Class 0)**

```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_baseline_test data --normal_class 0 --device cpu --hybrid False --thresholding fixed --pretrain True
```

**Example: Baseline Deep SVDD on MNIST (Normal Class 0)**

```bash
source myenv/bin/activate
python srcv2/main.py mnist mnist_LeNet log/mnist_baseline_test data --normal_class 0 --device cpu --hybrid False --thresholding fixed --pretrain True
```

**Example: Baseline Deep SVDD on CIFAR-10 (Normal Class 0)**

```bash
source myenv/bin/activate
python srcv2/main.py cifar10 cifar10_LeNet log/cifar10_baseline_test data --normal_class 0 --device cpu --hybrid False --thresholding fixed --pretrain True
```

### 2. Running New Hybrid Deep SVDD Models

To run the Hybrid Deep SVDD model, set `--hybrid True` and specify `mu1` and `mu2` for the weighting of SVDD and reconstruction losses. For comparison, keep `--thresholding fixed`.

**Example: Hybrid Deep SVDD on Fashion-MNIST (Normal Class 0, μ1=1.0, μ2=0.5)**

```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_hybrid_mu1_1_mu2_0.5_test data --normal_class 0 --device cpu --hybrid True --mu1 1.0 --mu2 0.5 --thresholding fixed --pretrain True
```

**Example: Hybrid Deep SVDD on MNIST (Normal Class 0, μ1=1.0, μ2=1.0)**

```bash
source myenv/bin/activate
python srcv2/main.py mnist mnist_LeNet log/mnist_hybrid_mu1_1_mu2_1.0_test data --normal_class 0 --device cpu --hybrid True --mu1 1.0 --mu2 1.0 --thresholding fixed --pretrain True
```

**Example: Hybrid Deep SVDD on CIFAR-10 (Normal Class 0, μ1=1.0, μ2=2.0)**

```bash
source myenv/bin/activate
python srcv2/main.py cifar10 cifar10_LeNet log/cifar10_hybrid_mu1_1_mu2_2.0_test data --normal_class 0 --device cpu --hybrid True --mu1 1.0 --mu2 2.0 --thresholding fixed --pretrain True
```

### 3. Running Adaptive Statistical Threshold Models

To run the Deep SVDD model with adaptive statistical thresholding, set `--hybrid False` and `--thresholding adaptive`.

**Example: Adaptive Threshold on Fashion-MNIST (Normal Class 0)**

```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_adaptive_threshold_test data --normal_class 0 --device cpu --hybrid False --thresholding adaptive --pretrain True
```

**Example: Adaptive Threshold on MNIST (Normal Class 0)**

```bash
source myenv/bin/activate
python srcv2/main.py mnist mnist_LeNet log/mnist_adaptive_threshold_test data --normal_class 0 --device cpu --hybrid False --thresholding adaptive --pretrain True
```

**Example: Adaptive Threshold on CIFAR-10 (Normal Class 0)**

```bash
source myenv/bin/activate
python srcv2/main.py cifar10 cifar10_LeNet log/cifar10_adaptive_threshold_test data --normal_class 0 --device cpu --hybrid False --thresholding adaptive --pretrain True
```

### 4. Generating Grad-CAM Visualizations

To generate Grad-CAM visualizations for any of the above models, add the `--grad_cam True` flag to your `srcv2/main.py` command. This will save `normal_grad_cam.png` and `outlier_grad_cam.png` in your `xp_path`.

**Example: Baseline Deep SVDD on Fashion-MNIST with Grad-CAM**

```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_baseline_test_grad_cam data --normal_class 0 --device cpu --hybrid False --thresholding fixed --pretrain True --grad_cam True
```

## Running the Original Baseline (from `src`)

The original baseline implementation can be run from the `src` directory.

### MNIST example
```bash
# activate virtual environment
source myenv/bin/activate

# create folder for experimental output
mkdir -p log/mnist_test

# change to source directory
cd src

# run experiment
python main.py mnist mnist_LeNet ../log/mnist_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3
```

### CIFAR-10 example
```bash
# activate virtual environment
source myenv/bin/activate

# create folder for experimental output
mkdir -p log/cifar10_test

# change to source directory
cd src

# run experiment
python main.py cifar10 cifar10_LeNet ../log/cifar10_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3
```

## 5. Analyzing Results and Deliverables

After running your experiments, the results (`results.json`, `log.txt`, and generated plots like `normals.png`, `outliers.png`, `normal_grad_cam.png`, `outlier_grad_cam.png`, `tsne_latent.png`) will be saved in the `xp_path` you specified (e.g., `log/my_experiment_name/`).

The `Deliverables/` folder contains:
*   `Deliverables/summary_table.md`: A Markdown table summarizing the AUC and F1 scores from various experiments.
*   `Deliverables/Plots/`: Organized subdirectories containing individual and comparative plots (ROC curves, t-SNE, Grad-CAM overlays) generated from the experiments.

You can use the provided Python scripts (`generate_roc_curves.py`, `generate_tsne_plots.py`, `generate_comparative_plots.py`) to regenerate or create comparative plots if needed, after ensuring the `srcv2` path is correctly added to your Python environment.
