"""2D Max Pooling Layer implementation."""

import numpy as np
from typing import Tuple, Union, Optional
from src.cnn.layers.base import Layer


class MaxPool2D(Layer):
    """2D Max Pooling Layer for spatial downsampling."""

    def __init__(
        self,
        pool_size: Union[int, Tuple[int, int]] = 2,
        stride: Optional[int] = None,
    ) -> None:
        """Initialize MaxPool2D layer.

        Args:
            pool_size: Spatial dimensions of pooling window (h, w) or integer.
            stride: Stride size. Defaults to pool_size if None.
        """
        super().__init__()
        self.trainable = False
        if isinstance(pool_size, int):
            self.ph, self.pw = pool_size, pool_size
        else:
            self.ph, self.pw = pool_size

        self.stride = stride if stride is not None else self.ph

        # Cache for backpropagation
        self.x: np.ndarray = np.array([])

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass for MaxPool2D.

        Args:
            inputs: Input tensor of shape (N, C, H, W) or (C, H, W).
            training: Whether layer is in training mode.

        Returns:
            Pooled output tensor of shape (N, C, H_out, W_out).
        """
        is_single = (inputs.ndim == 3)
        if is_single:
            inputs = np.expand_dims(inputs, axis=0)

        N, C, H, W = inputs.shape
        H_out = int((H - self.ph) / self.stride + 1)
        W_out = int((W - self.pw) / self.stride + 1)

        out = np.zeros((N, C, H_out, W_out), dtype=inputs.dtype)

        for i in range(H_out):
            for j in range(W_out):
                h_start, h_end = i * self.stride, i * self.stride + self.ph
                w_start, w_end = j * self.stride, j * self.stride + self.pw
                patch = inputs[:, :, h_start:h_end, w_start:w_end]
                out[:, :, i, j] = np.max(patch, axis=(2, 3))

        if training:
            self.x = inputs

        if is_single and not training:
            return out[0]
        return out

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Backward pass for MaxPool2D.

        Args:
            dL_dout: Gradient wrt output (N, C, H_out, W_out).

        Returns:
            Gradient wrt input (N, C, H, W).
        """
        if dL_dout.ndim == 3:
            dL_dout = np.expand_dims(dL_dout, axis=0)

        N, C, H, W = self.x.shape
        H_out, W_out = dL_dout.shape[2], dL_dout.shape[3]
        dL_dx = np.zeros_like(self.x)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start, h_end = i * self.stride, i * self.stride + self.ph
                        w_start, w_end = j * self.stride, j * self.stride + self.pw
                        patch = self.x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        mask = (patch == max_val)
                        dL_dx[n, c, h_start:h_end, w_start:w_end] += mask * dL_dout[n, c, i, j]

        return dL_dx
