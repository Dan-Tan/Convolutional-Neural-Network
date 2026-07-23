"""2D Convolutional Layer implementation using vectorized im2col operations."""

import numpy as np
from typing import Tuple, Union
from src.cnn.layers.base import Layer
from src.cnn.utils import im2col_indices, col2im_indices


class Conv2D(Layer):
    """2D Convolutional Layer with momentum-based stochastic gradient descent."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        """Initialize Conv2D layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of filters / output channels.
            kernel_size: Size of convolution kernel (height, width) or scalar.
            stride: Stride of convolution.
            padding: Zero-padding size.
            learning_rate: Learning rate for parameter updates.
            momentum: Momentum coefficient.
        """
        super().__init__()
        self.trainable = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kh, self.kw = kernel_size, kernel_size
        else:
            self.kh, self.kw = kernel_size

        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate
        self.momentum = momentum

        # He initialization for weights
        fan_in = in_channels * self.kh * self.kw
        self.W = np.random.randn(out_channels, fan_in).astype(np.float32) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros((out_channels, 1), dtype=np.float32)

        # Velocity accumulators for momentum
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

        # Cache for backpropagation
        self.x: np.ndarray = np.array([])
        self.x_col: np.ndarray = np.array([])

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass for Conv2D.

        Args:
            inputs: Array of shape (N, C, H, W) or (C, H, W).
            training: Whether layer is in training mode.

        Returns:
            Output tensor of shape (N, out_channels, H_out, W_out).
        """
        is_single = (inputs.ndim == 3)
        if is_single:
            inputs = np.expand_dims(inputs, axis=0)

        N, C, H, W = inputs.shape
        H_out = int((H + 2 * self.padding - self.kh) / self.stride + 1)
        W_out = int((W + 2 * self.padding - self.kw) / self.stride + 1)

        x_col = im2col_indices(inputs, self.kh, self.kw, self.padding, self.stride)

        # Linear convolution operation: out = W * X_col + b
        out = np.dot(self.W, x_col) + self.b
        out = out.reshape(self.out_channels, H_out, W_out, N)
        out = out.transpose(3, 0, 1, 2)

        if training:
            self.x = inputs
            self.x_col = x_col

        if is_single and not training:
            return out[0]
        return out

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Backward pass for Conv2D.

        Args:
            dL_dout: Gradient of loss wrt layer output (N, out_channels, H_out, W_out).

        Returns:
            Gradient of loss wrt layer input (N, in_channels, H, W).
        """
        if dL_dout.ndim == 3:
            dL_dout = np.expand_dims(dL_dout, axis=0)

        # Transpose dL_dout to (out_channels, H_out * W_out * N)
        dL_dout_reshaped = dL_dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

        # Gradients wrt parameters
        self.dW = np.dot(dL_dout_reshaped, self.x_col.T)
        self.db = np.sum(dL_dout_reshaped, axis=1, keepdims=True)

        # Gradient wrt input columns
        dL_dx_col = np.dot(self.W.T, dL_dout_reshaped)
        dL_dx = col2im_indices(dL_dx_col, self.x.shape, self.kh, self.kw, self.padding, self.stride)

        # Update momentum velocities & weights
        self.v_W = self.momentum * self.v_W + (1.0 - self.momentum) * self.dW
        self.v_b = self.momentum * self.v_b + (1.0 - self.momentum) * self.db

        self.W -= self.learning_rate * self.v_W
        self.b -= self.learning_rate * self.v_b

        return dL_dx
