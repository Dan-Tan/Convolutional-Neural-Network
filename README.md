# Convolutional Neural Network from Scratch in NumPy

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-013243.svg?logo=numpy)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-green.svg)](https://docs.pytest.org/)
[![Legacy Code](https://img.shields.io/badge/Legacy_Code-High_School_Project-orange.svg)](legacy/README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A 2D Convolutional Neural Network (CNN) framework implemented from first principles in NumPy, without relying on PyTorch or TensorFlow for model operations or automatic differentiation.

This project was originally created upon finishing high school as an early programming project to learn Python and understand neural network mathematics. The raw initial draft is archived in [legacy/original_cnn.py](legacy/original_cnn.py). The repository contains both that legacy script and a modernized implementation with modular layer abstractions, type annotations, unit tests, and vectorized `im2col` acceleration.

---

## Architecture Overview

```
Input Image (1, 28, 28)
       |
       v
 +-----------+     +-----------+     +-------------+
 |  Conv2D   | --> |   ReLU    | --> |  MaxPool2D  |  --> (8, 14, 14)
 | (8, 3x3)  |     | Activation|     |  (2x2, s=2) |
 +-----------+     +-----------+     +-------------+
       |
       v
 +-----------+     +-----------+     +-------------+
 |  Conv2D   | --> |   ReLU    | --> |  MaxPool2D  |  --> (16, 7, 7)
 | (16, 3x3) |     | Activation|     |  (2x2, s=2) |
 +-----------+     +-----------+     +-------------+
       |
       v
 +-----------+     +-----------+     +-------------+     +----------------------+
 |  Flatten  | --> |   Dense   | --> |    Dense    | --> | SoftmaxCrossEntropy  | --> Output (10)
 |   (784)   |     |   (64)    |     |    (10)     |     |      Loss            |
 +-----------+     +-----------+     +-------------+     +----------------------+
```

---

## Mathematical Foundations

### 1. Vectorized Convolution via im2col
Input patches are unrolled into columns to convert spatial 2D convolution into matrix multiplication:

$$\mathbf{X}_{\text{col}} \in \mathbb{R}^{(C_{\text{in}} \cdot K_h \cdot K_w) \times (N \cdot H_{\text{out}} \cdot W_{\text{out}})}$$

The forward pass is:

$$\mathbf{Y}_{\text{flat}} = \mathbf{W} \cdot \mathbf{X}_{\text{col}} + \mathbf{b}$$

Backpropagation gradients with respect to weights and input columns are derived via matrix transposes:

$$\frac{\partial L}{\partial \mathbf{W}} = \left(\frac{\partial L}{\partial \mathbf{Y}}\right) \cdot \mathbf{X}_{\text{col}}^T, \quad \frac{\partial L}{\partial \mathbf{X}_{\text{col}}} = \mathbf{W}^T \cdot \left(\frac{\partial L}{\partial \mathbf{Y}}\right)$$

### 2. Softmax + Cross-Entropy Fused Gradient
For Softmax outputs $\hat{y}_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}$ and Cross-Entropy loss $L = -\sum_k y_k \log \hat{y}_k$, the gradient with respect to pre-activation logits is:

$$\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}$$

---

## Visualizations

### Intermediate Feature Maps
Visualizing layer outputs shows how convolution filters detect edge primitives and feature boundaries:

| Layer 1 Feature Maps (16 Filters) | Layer 2 Feature Maps (32 Filters) |
| :---: | :---: |
| ![Layer 1 Feature Maps](assets/feature_maps_layer1.png) | ![Layer 2 Feature Maps](assets/feature_maps_layer2.png) |

### Training Convergence
![Training Curves](assets/training_curves.png)

---

## Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/Dan-Tan/Convolutional-Neural-Network.git
cd Convolutional-Neural-Network

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the CNN
Run the training pipeline on MNIST:

```bash
python train.py --epochs 5 --batch-size 64 --lr 0.01
```

### 3. Generate Visualizations
Extract feature maps and generate plots in `assets/`:

```bash
python visualize.py
```

### 4. Run Tests
Run the test suite using `pytest`:

```bash
PYTHONPATH=. pytest
```

---

## License
This project is open-source under the [MIT License](LICENSE).
