"""Activation functions and loss layers."""

import numpy as np
from src.cnn.layers.base import Layer


class ReLU(Layer):
    """Rectified Linear Unit activation layer."""

    def __init__(self) -> None:
        super().__init__()
        self.trainable = False
        self.x: np.ndarray = np.array([])

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute ReLU activation out = max(0, x)."""
        if training:
            self.x = inputs
        return np.maximum(0.0, inputs)

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Compute gradient wrt input."""
        return dL_dout * (self.x > 0.0).astype(np.float32)


class LeakyReLU(Layer):
    """Leaky Rectified Linear Unit activation layer."""

    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.trainable = False
        self.alpha = alpha
        self.x: np.ndarray = np.array([])

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute LeakyReLU activation."""
        if training:
            self.x = inputs
        return np.where(inputs > 0.0, inputs, self.alpha * inputs)

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Compute gradient wrt input."""
        dx = np.ones_like(self.x)
        dx[self.x <= 0.0] = self.alpha
        return dL_dout * dx


class SoftmaxCrossEntropyLoss:
    """Softmax Activation combined with Cross-Entropy Loss for numerical stability."""

    def __init__(self) -> None:
        self.y_hat: np.ndarray = np.array([])
        self.targets: np.ndarray = np.array([])

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Compute stable Softmax probabilities and Cross-Entropy Loss.

        Args:
            logits: Logits matrix (N, num_classes).
            targets: One-hot ground truth matrix (N, num_classes) or integer class indices.

        Returns:
            Scalar average cross-entropy loss.
        """
        # Ensure 2D
        if logits.ndim == 1:
            logits = np.expand_dims(logits, axis=0)

        # Numerical stability shift
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        self.y_hat = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        if targets.ndim == 1:
            N = targets.shape[0]
            self.targets = np.zeros_like(self.y_hat)
            self.targets[np.arange(N), targets] = 1.0
        else:
            self.targets = targets

        # Cross-Entropy Loss with epsilon clip
        eps = 1e-12
        probs_clipped = np.clip(self.y_hat, eps, 1.0 - eps)
        loss = -np.sum(self.targets * np.log(probs_clipped)) / logits.shape[0]
        return float(loss)

    def backward(self) -> np.ndarray:
        """Compute gradient of Softmax + Cross Entropy loss wrt logits dL/dz = y_hat - y."""
        N = self.y_hat.shape[0]
        return (self.y_hat - self.targets) / N
