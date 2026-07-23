"""Flatten layer to reshape multi-dimensional feature maps into 1D vectors."""

import numpy as np
from typing import Tuple
from src.cnn.layers.base import Layer


class Flatten(Layer):
    """Flattens input tensor dimensions (N, C, H, W) to (N, C * H * W)."""

    def __init__(self) -> None:
        super().__init__()
        self.trainable = False
        self.orig_shape: Tuple[int, ...] = ()

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Flatten spatial dimensions.

        Args:
            inputs: Input tensor of shape (N, C, H, W) or (C, H, W).
            training: Whether layer is in training mode.

        Returns:
            2D array of shape (N, C * H * W).
        """
        if inputs.ndim == 3:
            inputs = np.expand_dims(inputs, axis=0)

        if training:
            self.orig_shape = inputs.shape

        N = inputs.shape[0]
        return inputs.reshape(N, -1)

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Reshape gradient back to original input tensor shape.

        Args:
            dL_dout: Gradient array of shape (N, feature_dim).

        Returns:
            Reshaped gradient array matching original input shape.
        """
        return dL_dout.reshape(self.orig_shape)
