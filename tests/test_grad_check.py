"""Unit tests for numerical gradient checking across CNN layers."""

import pytest
import numpy as np
from src.cnn.layers.conv import Conv2D
from src.cnn.layers.dense import Dense
from src.cnn.layers.activations import SoftmaxCrossEntropyLoss
from src.cnn.grad_check import eval_numerical_gradient, compute_relative_error


def test_dense_gradient_check():
    """Verify analytical vs numerical gradient for Dense layer weights."""
    np.random.seed(42)
    N, D_in, D_out = 2, 4, 3
    x = np.random.randn(N, D_in).astype(np.float64)
    y_true = np.array([0, 2])

    dense = Dense(D_in, D_out)
    dense.W = dense.W.astype(np.float64)
    dense.b = dense.b.astype(np.float64)
    loss_fn = SoftmaxCrossEntropyLoss()

    def loss_func(W_param):
        orig_W = dense.W
        dense.W = W_param
        logits = dense.forward(x, training=False)
        loss = loss_fn.forward(logits, y_true)
        dense.W = orig_W
        return loss

    W_init = dense.W.copy()

    # Forward & backward pass
    logits = dense.forward(x, training=True)
    loss_fn.forward(logits, y_true)
    dL_dlogits = loss_fn.backward()

    dense.backward(dL_dlogits)
    analytical_grad_W = dense.dW

    # Reset W to initial value for numerical evaluation
    dense.W = W_init.copy()
    numerical_grad_W = eval_numerical_gradient(loss_func, W_init.copy(), h=1e-5)

    error = compute_relative_error(analytical_grad_W, numerical_grad_W)
    assert error < 1e-4, f"Dense weight relative error too high: {error}"


def test_conv2d_gradient_check():
    """Verify analytical vs numerical gradient for Conv2D layer weights."""
    np.random.seed(42)
    N, C_in, H, W_dim = 2, 1, 6, 6
    out_channels = 2
    kernel_size = 3

    x = np.random.randn(N, C_in, H, W_dim).astype(np.float64)
    conv = Conv2D(in_channels=C_in, out_channels=out_channels, kernel_size=kernel_size, padding=1)
    conv.W = conv.W.astype(np.float64)
    conv.b = conv.b.astype(np.float64)

    def loss_func(W_param):
        orig_W = conv.W
        conv.W = W_param
        out = conv.forward(x, training=False)
        loss = float(np.sum(out ** 2))
        conv.W = orig_W
        return loss

    W_init = conv.W.copy()

    out = conv.forward(x, training=True)
    dL_dout = 2.0 * out  # Derivative of sum(out^2)

    conv.backward(dL_dout)
    analytical_grad_W = conv.dW

    conv.W = W_init.copy()
    numerical_grad_W = eval_numerical_gradient(loss_func, W_init.copy(), h=1e-6)

    error = compute_relative_error(analytical_grad_W, numerical_grad_W)
    assert error < 1e-2, f"Conv2D weight relative error too high: {error}"
