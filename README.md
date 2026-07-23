# Convolutional Neural Network from Scratch in NumPy

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/Library-NumPy-013243.svg?logo=numpy)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-green.svg)](https://docs.pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, modular 2D Convolutional Neural Network (CNN) framework implemented from first principles in **NumPy**, without relying on PyTorch or TensorFlow for model operations or automatic differentiation.

> 💡 **Project Provenance & Retrospective**  
> This project originally started when I finished high school as one of my earliest programming projects while learning Python and exploring machine learning mathematics from scratch.
> 
> To preserve the historical context of learning to code, the raw, unedited initial draft is retained in **[`legacy/original_cnn.py`](file:///home/dtan/projects/Convolutional-Neural-Network/legacy/README.md)**. This repository now presents both that original implementation and a modernized, clean refactor (`src/`) featuring modular layer abstraction, type hints, unit tests, and vectorized `im2col` acceleration.

---

## 🌟 Key Features

- **First-Principles Engine**: Core layers (`Conv2D`, `MaxPool2D`, `Dense`, `Flatten`, `ReLU`, `Softmax`) built entirely with raw NumPy matrix operations.
- **Vectorized Convolutions**: Implements `im2col` and `col2im` transformation to turn spatial 2D convolutions into fast matrix multiplications.
- **Modular Sequential API**: Clean `Sequential` container supporting `.fit()`, `.evaluate()`, and `.predict()`.
- **Numerically Stable Backpropagation**: Fused Softmax with Cross-Entropy Loss to handle gradient propagation safely.
- **Visualization Suite**: Scripts to extract intermediate feature maps and plot loss/accuracy training curves.
- **Automated Test Suite**: Unit tests powered by `pytest` covering layer shapes, activation maps, and end-to-end integration.

---

## 🏗️ Architecture Overview

```
Input Image (1, 28, 28)
       │
       ▼
 ┌───────────┐     ┌───────────┐     ┌─────────────┐
 │  Conv2D   │ ──► │   ReLU    │ ──► │  MaxPool2D  │  ──► (8, 14, 14)
 │ (8, 3x3)  │     │ Activation│     │  (2x2, s=2) │
 └───────────┘     └───────────┘     └─────────────┘
       │
       ▼
 ┌───────────┐     ┌───────────┐     ┌─────────────┐
 │  Conv2D   │ ──► │   ReLU    │ ──► │  MaxPool2D  │  ──► (16, 7, 7)
 │ (16, 3x3) │     │ Activation│     │  (2x2, s=2) │
 └───────────┘     └───────────┘     └─────────────┘
       │
       ▼
 ┌───────────┐     ┌───────────┐     ┌─────────────┐     ┌──────────────────────┐
 │  Flatten  │ ──► │   Dense   │ ──► │    Dense    │ ──► │ SoftmaxCrossEntropy  │ ──► Output (10)
 │   (784)   │     │   (64)    │     │    (10)     │     │      Loss            │
 └───────────┘     └───────────┘     └─────────────┘     └──────────────────────┘
```

---

## 📐 Mathematical Foundations

### 1. Vectorized Convolution via `im2col`
Instead of using slow nested loops over height, width, and channels, input patches are unrolled into columns:

$$\mathbf{X}_{\text{col}} \in \mathbb{R}^{(C_{\text{in}} \cdot K_h \cdot K_w) \times (N \cdot H_{\text{out}} \cdot W_{\text{out}})}$$

The forward convolution reduces to a single matrix multiplication:

$$\mathbf{Y}_{\text{flat}} = \mathbf{W} \cdot \mathbf{X}_{\text{col}} + \mathbf{b}$$

During backpropagation, gradients wrt weights and input columns are derived via matrix transposes:

$$\frac{\partial L}{\partial \mathbf{W}} = \left(\frac{\partial L}{\partial \mathbf{Y}}\right) \cdot \mathbf{X}_{\text{col}}^T, \quad \frac{\partial L}{\partial \mathbf{X}_{\text{col}}} = \mathbf{W}^T \cdot \left(\frac{\partial L}{\partial \mathbf{Y}}\right)$$

### 2. Softmax + Cross-Entropy Fused Gradient
For Softmax outputs $\hat{y}_k = \frac{e^{z_k}}{\sum_{j} e^{z_j}}$ and Cross-Entropy loss $L = -\sum_k y_k \log \hat{y}_k$, the gradient with respect to pre-activation logits simplifies to:

$$\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}$$

---

## 📊 Visualizations

### Intermediate Feature Maps
Visualizing layer outputs highlights how early convolution filters detect edge primitives and feature boundaries:

| Layer 1 Feature Maps (16 Filters) | Layer 2 Feature Maps (32 Filters) |
| :---: | :---: |
| ![Layer 1 Feature Maps](assets/feature_maps_layer1.png) | ![Layer 2 Feature Maps](assets/feature_maps_layer2.png) |

### Training Convergence
![Training Curves](assets/training_curves.png)

---

## 🔄 Codebase Evolution: Initial Draft vs. Modernized Architecture

| Component | Initial Draft (`legacy/original_cnn.py`) | Refactored Architecture (`src/`) |
| :--- | :--- | :--- |
| **Structure** | Monolithic script exported from Google Colab. | Modular Python package with decoupled layer objects. |
| **Array Safety** | In-place boolean mutations (`z[z < 0] = 0`). | Immutable array ops with explicit gradient caching. |
| **Data Ingestion** | Hardcoded file paths (`/content/drive/...`). | Automated dataset downloader with synthetic fallback. |
| **Code Style** | Basic script without type annotations or docstrings. | PEP-8 compliant with PEP-484 type annotations and Google docstrings. |
| **Verification** | Manual inline output printing. | Automated test suite powered by `pytest`. |

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/Dan-Tan/Convolutional-Neural-Network.git
cd Convolutional-Neural-Network

# Create and activate virtual environment
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

## 📂 Project Structure

```
Convolutional-Neural-Network/
├── assets/                       # Generated feature maps & training plots
├── legacy/
│   ├── original_cnn.py           # Unedited initial implementation
│   └── README.md                 # Notes on original implementation
├── src/
│   └── cnn/
│       ├── __init__.py           # Package exports
│       ├── dataset.py            # Dataset loader & synthetic fallback
│       ├── network.py            # Sequential model container
│       ├── utils.py              # im2col/col2im vectorization helpers
│       └── layers/
│           ├── base.py           # Abstract Base Layer
│           ├── conv.py           # 2D Convolution layer (im2col)
│           ├── pool.py           # 2D Max Pooling layer
│           ├── dense.py          # Fully Connected layer
│           ├── flatten.py        # Dimension flattener
│           └── activations.py    # ReLU, LeakyReLU, SoftmaxCrossEntropy
├── tests/                        # Pytest test suite
│   ├── test_layers.py
│   └── test_network.py
├── cnn.py                        # Root entry point script
├── train.py                      # Training CLI script
├── visualize.py                  # Visualization script
├── pyproject.toml                # Package configuration
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```
