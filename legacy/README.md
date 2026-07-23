# Legacy Implementation (High School Project)

This directory preserves the unedited Python script (`original_cnn.py`) written upon finishing high school as one of my first hands-on programming projects exploring deep learning mathematics from first principles.

The exact original commit prior to codebase refactoring is tagged as [`v0.1.0-legacy`](https://github.com/Dan-Tan/Convolutional-Neural-Network/tree/v0.1.0-legacy) (commit [`5cdbe1fa7923f47897f59852014a45b88c0fcef8`](https://github.com/Dan-Tan/Convolutional-Neural-Network/commit/5cdbe1fa7923f47897f59852014a45b88c0fcef8)).

## Background & Provenance
* **Original File**: Exported from a Google Colaboratory notebook (`CNN.ipynb`).
* **Framework Constraints**: Written entirely from scratch using raw **NumPy** matrix operations, without relying on PyTorch or TensorFlow for model execution or backpropagation. (TensorFlow/Keras was used solely to download the raw MNIST dataset array).
* **Early Milestones**:
  * Formulated manual 2D convolution vectorization via `im2col`-like matrix unrolling.
  * Derived manual backpropagation equations for 2D Convolutional layers, Max Pooling layers, Dense layers, ReLU activations, and Softmax output layers.
  * Implemented SGD with momentum and learning rate decay schedules.

---
*The original script is preserved unchanged in `original_cnn.py` as well as in the git tag `v0.1.0-legacy`.*
