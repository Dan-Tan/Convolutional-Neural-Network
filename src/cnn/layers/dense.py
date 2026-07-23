"""Fully Connected (Dense) Layer implementation."""

import numpy as np
from src.cnn.layers.base import Layer


class Dense(Layer):
    """Dense / Fully Connected Layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
    ) -> None:
        """Initialize Dense layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features / units.
            learning_rate: Learning rate for SGD updates.
            momentum: Momentum coefficient.
        """
        super().__init__()
        self.trainable = True
        self.in_features = in_features
        self.out_features = out_features
        self.learning_rate = learning_rate
        self.momentum = momentum

        # He / Xavier normal initialization
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features), dtype=np.float32)

        # Velocity accumulators for momentum
        self.v_W = np.zeros_like(self.W)
        self.v_b = np.zeros_like(self.b)

        # Cache for backpropagation
        self.x: np.ndarray = np.array([])

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass for Dense layer: Y = X * W + b.

        Args:
            inputs: Array of shape (N, in_features) or (in_features,).
            training: Whether layer is in training mode.

        Returns:
            Output array of shape (N, out_features).
        """
        is_single = (inputs.ndim == 1)
        if is_single:
            inputs = np.expand_dims(inputs, axis=0)

        out = np.dot(inputs, self.W) + self.b

        if training:
            self.x = inputs

        if is_single and not training:
            return out[0]
        return out

    def backward(self, dL_dout: np.ndarray) -> np.ndarray:
        """Backward pass for Dense layer.

        Args:
            dL_dout: Gradient wrt output (N, out_features).

        Returns:
            Gradient wrt input (N, in_features).
        """
        if dL_dout.ndim == 1:
            dL_dout = np.expand_dims(dL_dout, axis=0)

        N = self.x.shape[0]
        dL_dW = np.dot(self.x.T, dL_dout) / N
        dL_db = np.sum(dL_dout, axis=0, keepdims=True) / N

        dL_dx = np.dot(dL_dout, self.W.T)

        # Update velocities & parameters
        self.v_W = self.momentum * self.v_W + (1.0 - self.momentum) * dL_dW
        self.v_b = self.momentum * self.v_b + (1.0 - self.momentum) * dL_db

        self.W -= self.learning_rate * self.v_W
        self.b -= self.learning_rate * self.v_b

        return dL_dx
