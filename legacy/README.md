# Legacy Implementation (High School Project)

This directory preserves the unedited Python script (`original_cnn.py`) written upon finishing high school as one of my first hands-on programming projects exploring deep learning mathematics from first principles.

## Background & Provenance
* **Original File**: Exported from a Google Colaboratory notebook (`CNN.ipynb`).
* **Framework Constraints**: Written entirely from scratch using raw **NumPy** matrix operations, without relying on PyTorch or TensorFlow for model execution or backpropagation. (TensorFlow/Keras was used solely to download the raw MNIST dataset array).
* **Early Milestones**:
  * Formulated manual 2D convolution vectorization via `im2col`-like matrix unrolling.
  * Derived manual backpropagation equations for 2D Convolutional layers, Max Pooling layers, Dense layers, ReLU activations, and Softmax output layers.
  * Implemented SGD with momentum and learning rate decay schedules.

## Retrospective & Codebase Evolution

| Technical Aspect | Initial Draft (`original_cnn.py`) | Refactored Suite (`src/`) |
| :--- | :--- | :--- |
| **Architecture** | Single monolithic script (548 lines) mixing dataset handling, layer logic, and execution code. | Modular OOP architecture (`Conv2D`, `MaxPool2D`, `Dense`, `Sequential`) adhering to modern conventions. |
| **Array Safety** | In-place boolean mutations (`z[z < 0] = 0`) causing state side-effects in backward passes. | Immutable array calculations and explicit gradient caching. |
| **Data Pipelines** | Hardcoded Google Drive paths (`/content/drive/...`) and channel duplication workarounds. | Modular dataset loader with automatic HTTP download fallback & synthetic fallback. |
| **Numerical Stability**| Manual cross-entropy & Softmax gradient calculations susceptible to zero-division underflows. | Numerically stable log-sum-exp Softmax with fused Cross-Entropy backpropagation ($\frac{\partial L}{\partial z} = \hat{y} - y$). |
| **Testing & Quality** | Manual inline `print()` outputs without automated test assertions. | Automated `pytest` test suite covering layer shapes, activations, and integration. |

---
*The original script is preserved unchanged in `original_cnn.py` for historical reference.*
