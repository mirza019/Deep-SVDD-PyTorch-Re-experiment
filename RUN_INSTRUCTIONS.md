# Running Experiments: Baseline vs. Hybrid Models

This document provides clear instructions on how to run the original baseline Deep SVDD models and the new hybrid/re-producibility experiments.

**Key Distinction:**
*   Use the `src` directory to run the **original baseline** model as described in the paper.
*   Use the `srcv2` directory to run all **new, extended experiments**, including the hybrid model, adaptive thresholding, Grad-CAM visualizations, and baseline comparisons.

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

## Running Baseline Experiments (Original `src` Implementation)

To run the original Deep SVDD model as implemented in the `src` directory, follow these examples.

### MNIST Example
```bash
# Activate virtual environment
source myenv/bin/activate

# Create folder for experimental output
mkdir -p log/mnist_test

# Change to the original source directory
cd src

# Run experiment
python main.py mnist mnist_LeNet ../log/mnist_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 3
```

### CIFAR-10 Example
```bash
# Activate virtual environment
source myenv/bin/activate

# Create folder for experimental output
mkdir -p log/cifar10_test

# Change to the original source directory
cd src

# Run experiment
python main.py cifar10 cifar10_LeNet ../log/cifar10_test ../data --objective one-class --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 3
```

---

## Running Hybrid & Re-experiments (`srcv2` Implementation)

All new and extended experiments are executed using the `srcv2/main.py` script. This script allows you to configure the hybrid loss, adaptive thresholding, and Grad-CAM visualizations.

### Common Parameters for `srcv2`:
*   `dataset_name`: `mnist`, `fashion_mnist`, `cifar10`
*   `net_name`: `mnist_LeNet`, `cifar10_LeNet`, `cifar10_LeNet_ELU`
*   `xp_path`: Path to save experiment logs and results
*   `data_path`: Root path for datasets
*   `--normal_class`: The class to be treated as normal
*   `--device`: Computation device (e.g., `cpu`)
*   `--pretrain`: Whether to pretrain the network

### 1. Re-running Baseline with `srcv2`

**Example: Baseline on Fashion-MNIST**
```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_baseline_test data --normal_class 0 --device cpu --hybrid False --thresholding fixed --pretrain True
```

### 2. Running Hybrid Deep SVDD Models

**Example: Hybrid on Fashion-MNIST (μ1=1.0, μ2=0.5)**
```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_hybrid_mu1_1_mu2_0.5_test data --normal_class 0 --device cpu --hybrid True --mu1 1.0 --mu2 0.5 --thresholding fixed --pretrain True
```

### 3. Running Adaptive Threshold Models

**Example: Adaptive Threshold on Fashion-MNIST**
```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_adaptive_threshold_test data --normal_class 0 --device cpu --hybrid False --thresholding adaptive --pretrain True
```

### 4. Generating Grad-CAM Visualizations

**Example: Baseline on Fashion-MNIST with Grad-CAM**
```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_baseline_test_grad_cam data --normal_class 0 --device cpu --hybrid False --thresholding fixed --pretrain True --grad_cam True
```

## Analyzing Results

After running experiments, results and plots will be saved in the specified `xp_path`. The `Deliverables/` folder contains summaries and comparative plots from the re-experimentation work.