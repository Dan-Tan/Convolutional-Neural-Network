"""Base layer class for neural network layers."""

from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """Abstract base class for all neural network layers."""

    def __init__(self) -> None:
        self.trainable: bool = False

    @abstractmethod
    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Compute the forward pass.

        Args:
            inputs: Input tensor.
            training: Whether in training or evaluation mode.

        Returns:
            Output tensor.
        """
        pass

    @abstractmethod
    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Compute the backward pass (gradient propagation).

        Args:
            dL_dout: Gradient of the loss with respect to output of this layer.

        Returns:
            Gradient of the loss with respect to input of this layer.
        """
        pass

    def update_params(self, lr: float, momentum: float = 0.0) -> None:
        """Update layer parameters (if trainable)."""
        pass
