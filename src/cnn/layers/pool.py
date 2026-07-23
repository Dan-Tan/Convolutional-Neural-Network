"""2D Max Pooling Layer implementation using vectorized im2col operations."""

import numpy as np
from typing import Tuple, Union, Optional
from src.cnn.layers.base import Layer
from src.cnn.utils import im2col_indices, col2im_indices


class MaxPool2D(Layer):
    """2D Max Pooling Layer for spatial downsampling with vectorized indexing."""

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
        self.x_col: np.ndarray = np.array([])
        self.max_idx: np.ndarray = np.array([])

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Vectorized forward pass for MaxPool2D without Python loops.

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

        # Reshape input to treat channels as additional batch dimension
        x_reshaped = inputs.reshape(N * C, 1, H, W)
        x_col = im2col_indices(x_reshaped, self.ph, self.pw, padding=0, stride=self.stride)

        # Compute max elements and argmax indices along column axis
        max_idx = np.argmax(x_col, axis=0)
        out = np.max(x_col, axis=0)

        out = out.reshape(H_out, W_out, N, C).transpose(2, 3, 0, 1)

        if training:
            self.x = inputs
            self.x_col = x_col
            self.max_idx = max_idx

        if is_single and not training:
            return out[0]
        return out

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Vectorized backward pass for MaxPool2D routing gradients to max indices.

        Args:
            dL_dout: Gradient wrt output (N, C, H_out, W_out).

        Returns:
            Gradient wrt input (N, C, H, W).
        """
        if dL_dout.ndim == 3:
            dL_dout = np.expand_dims(dL_dout, axis=0)

        N, C, H, W = self.x.shape
        dL_dx_col = np.zeros_like(self.x_col)

        # Flatten dL_dout to match column layout
        dL_dout_flat = dL_dout.transpose(2, 3, 0, 1).ravel()

        # Direct vector assignment to argmax locations
        dL_dx_col[self.max_idx, np.arange(self.max_idx.size)] = dL_dout_flat

        # Transform column gradient back to 4D image gradient
        dL_dx = col2im_indices(dL_dx_col, (N * C, 1, H, W), self.ph, self.pw, padding=0, stride=self.stride)
        dL_dx = dL_dx.reshape(N, C, H, W)

        return dL_dx
