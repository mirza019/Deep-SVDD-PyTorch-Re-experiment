# Hybrid Explainable One-Class Framework for Deep Anomaly Detection

This repository contains a PyTorch implementation of "Deep One-Class Classification" (Deep SVDD) and extends it with a **Hybrid Explainable One-Class Framework**. This framework enhances the original Deep SVDD by integrating a hybrid loss function, adaptive statistical thresholding, and Grad-CAM for improved accuracy and interpretability.

This work is based on the original paper by Ruff et al. (ICML 2018).

## A Reader's Summary of the Core Paper

"Okay, so after looking through the 'Deep One-Class Classification' paper, here's what I've gathered.

My understanding is that the authors are tackling the problem of anomaly detection, especially for complex data like images where traditional methods don't do so well. They point out that while other deep learning methods exist, they are often repurposed from other tasks, like autoencoders that measure reconstruction error, rather than being designed specifically for anomaly detection.

The core idea they propose is a new method called **Deep Support Vector Data Description (Deep SVDD)**. The whole approach is pretty clever. It works by training a neural network on **only the 'normal' data**. The network's main goal isn't to classify things, but to learn a representation of the data. Specifically, it tries to map all the normal data points into the smallest possible sphere (a 'hypersphere') in a feature space. This forces the network to learn the essential, common characteristics of the normal class.

Detection happens when you test a new, unseen data point. You feed it to the trained network, and if the output lands inside or very close to that sphere, it's considered normal. If it lands far away, it's flagged as an anomaly. The distance from the center of the sphere is basically the anomaly score.

To evaluate their method, they adapted standard multi-class datasets like MNIST and CIFAR-10 for one-class classification. Here's a clearer breakdown of how they treated the data for each experiment:

*   **Defining 'Normal' vs. 'Anomalous'**: First, they chose one specific class from the dataset to be "normal" (for example, the digit '3'). All other classes (digits '0', '1', '2', '4', etc.) were then considered "anomalous" for that experiment.

*   **Training Phase**: The network was trained **exclusively** on images from the "normal" class (e.g., only images of '3'). It never saw a single image from the other nine "anomalous" classes during this phase.

*   **Testing Phase**: After training, the model was evaluated on a test set that contained a mix of images from **all ten classes**. The goal was to see if the model, having only ever seen the "normal" class, could correctly identify images from that class while rejecting images from the nine "anomalous" classes it had never encountered.

Finally, the paper shows that this Deep SVDD method performs really well, often outperforming both the classic methods and the other deep learning techniques in their experiments. It seems like a more direct and effective way to train a network for one-class classification."

## Key Features and Extensions

*   **Hybrid Deep SVDD:** Combines the Deep SVDD loss with an autoencoder's reconstruction loss to improve robustness and accuracy, especially for complex data distributions.
*   **Adaptive Statistical Thresholding:** Replaces the fixed hypersphere radius with a Mahalanobis distance-based adaptive threshold, leading to fewer false positives and better generalization.
*   **Grad-CAM for Interpretability:** Integrates Grad-CAM to provide visual explanations for why a sample is flagged as an anomaly, highlighting the most influential regions in an image.
*   **Refactored Codebase (`srcv2`):** A modernized and modular codebase (`srcv2`) that is more flexible, configurable, and easier to extend.

## Comparative Results Summary

The following table summarizes the AUC and F1 scores from various experiments, comparing the baseline Deep SVDD with our hybrid and adaptive thresholding models.

| Experiment | AUC | F1 Score |
|---|---|---|
| Fashion-MNIST Baseline | 0.9092 | 0.9634 |
| Fashion-MNIST Hybrid (mu1=1.0, mu2=0.5) | 0.9135 | 0.9633 |
| Fashion-MNIST Adaptive Threshold | 0.8865 | 0.9561 |
| MNIST Baseline | 0.9814 | 0.9825 |
| MNIST Hybrid (mu1=1.0, mu2=1.0) | 0.9863 | 0.9879 |
| MNIST Adaptive Threshold | 0.9848 | 0.9858 |
| CIFAR-10 Baseline | 0.6074 | 0.9474 |
| CIFAR-10 Hybrid (mu1=1.0, mu2=2.0) | 0.6306 | 0.9474 |
| CIFAR-10 Adaptive Threshold | 0.6142 | 0.9474 |

## Installation

This code is written in `Python 3.x` and requires the packages listed in `requirements.txt`.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mirza019/Deep-SVDD-PyTorch-Re-experiment.git
    cd Deep-SVDD-PyTorch-Re-experiment
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python3 -m venv myenv
    source myenv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running Experiments

This project contains two separate implementations:
*   **`src/`**: The original implementation, ideal for running the baseline model.
*   **`srcv2/`**: An updated and extended implementation for running hybrid models, adaptive thresholding, and other re-experiments.

**A Note on the Experiments:** In our experiments, we evaluate the model's ability to perform one-class classification. For a given dataset like MNIST or CIFAR-10, we designate one class as "normal" (e.g., the digit '3', using the `--normal_class` parameter) and treat all other classes as "anomalous." The model is then trained exclusively on data from the "normal" class. This process is repeated for each class to ensure a comprehensive evaluation.

---

## Running Baseline Experiments (Original `src` Implementation)

To run the original Deep SVDD model, use the `main.py` script in the `src` directory.

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

---

## Running Hybrid & Re-experiments (`srcv2` Implementation)

All new and extended experiments are executed using the `srcv2/main.py` script.

### 1. Re-running Baseline with `srcv2`
```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_baseline_test data --normal_class 0 --device cpu --hybrid False --thresholding fixed --pretrain True
```

### 2. Running Hybrid Deep SVDD Models
```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_hybrid_mu1_1_mu2_0.5_test data --normal_class 0 --device cpu --hybrid True --mu1 1.0 --mu2 0.5 --thresholding fixed --pretrain True
```

### 3. Generating Grad-CAM Visualizations
```bash
source myenv/bin/activate
python srcv2/main.py fashion_mnist mnist_LeNet log/fashion_mnist_baseline_test_grad_cam data --normal_class 0 --device cpu --hybrid False --thresholding fixed --pretrain True --grad_cam True
```

## Deliverables

The `Deliverables/` folder contains the final results of the experiments, including plots and a summary table.

*   **`Deliverables/summary_table.md`**: A Markdown table summarizing the AUC and F1 scores from all experiments.

*   **`Deliverables/Plots/`**: This directory contains all the plots generated from the experiments, organized into the following subdirectories:
    *   **`CIFAR10/`**: Plots from experiments run on the CIFAR-10 dataset.
    *   **`FashionMNIST/`**: Plots from experiments run on the Fashion-MNIST dataset.
    *   **`MNIST/`**: Plots from experiments run on the MNIST dataset.
    *   **`Comparative/`**: Plots that compare the performance of different models (e.g., ROC curves).

## Original Abstract (Deep One-Class Classification)
> Despite the great advances made by deep learning in many machine learning problems, there is a relative dearth of deep learning approaches for anomaly detection. Those approaches which do exist involve networks trained to perform a task other than anomaly detection, namely generative models or compression, which are in turn adapted for use in anomaly detection; they are not trained on an anomaly detection based objective. In this paper we introduce a new anomaly detection method—Deep Support Vector Data Description—, which is trained on an anomaly detection based objective. The adaptation to the deep regime necessitates that our neural network and training procedure satisfy certain properties, which we demonstrate theoretically. We show the effectiveness of our method on MNIST and CIFAR-10 image benchmark datasets as well as on the detection of adversarial examples of GTSRB stop signs.

## Citation
If you use this work, please cite the original paper:
```
@InProceedings{pmlr-v80-ruff18a,
  title     = {Deep One-Class Classification},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Deecke, Lucas and Siddiqui, Shoaib A. and Binder, Alexander and M{\"u}ller, Emmanuel and Kloft, Marius},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages     = {4393--4402},
  year      = {2018},
  volume    = {80},
}
```

## License
MIT
