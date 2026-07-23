"""Unit tests for CNN layers."""

import pytest
import numpy as np
from src.cnn.layers.conv import Conv2D
from src.cnn.layers.pool import MaxPool2D
from src.cnn.layers.dense import Dense
from src.cnn.layers.flatten import Flatten
from src.cnn.layers.activations import ReLU, LeakyReLU, SoftmaxCrossEntropyLoss


def test_conv2d_shape():
    batch_size = 4
    in_c, out_c = 1, 8
    h, w = 28, 28
    kernel_size = 3
    padding = 1
    stride = 1

    x = np.random.randn(batch_size, in_c, h, w).astype(np.float32)
    conv = Conv2D(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size, padding=padding, stride=stride)

    out = conv.forward(x, training=True)
    assert out.shape == (batch_size, out_c, 28, 28)

    dL_dout = np.random.randn(*out.shape).astype(np.float32)
    dL_dx = conv.backward(dL_dout)
    assert dL_dx.shape == x.shape


def test_maxpool2d_shape():
    batch_size = 4
    c, h, w = 8, 28, 28
    pool = MaxPool2D(pool_size=2, stride=2)

    x = np.random.randn(batch_size, c, h, w).astype(np.float32)
    out = pool.forward(x, training=True)
    assert out.shape == (batch_size, c, 14, 14)

    dL_dout = np.random.randn(*out.shape).astype(np.float32)
    dL_dx = pool.backward(dL_dout)
    assert dL_dx.shape == x.shape


def test_flatten_shape():
    x = np.random.randn(4, 8, 14, 14).astype(np.float32)
    flatten = Flatten()
    out = flatten.forward(x, training=True)
    assert out.shape == (4, 8 * 14 * 14)

    dL_dx = flatten.backward(out)
    assert dL_dx.shape == x.shape


def test_dense_shape():
    x = np.random.randn(4, 128).astype(np.float32)
    dense = Dense(in_features=128, out_features=10)

    out = dense.forward(x, training=True)
    assert out.shape == (4, 10)

    dL_dout = np.random.randn(*out.shape).astype(np.float32)
    dL_dx = dense.backward(dL_dout)
    assert dL_dx.shape == x.shape


def test_relu_activation():
    relu = ReLU()
    x = np.array([[-1.0, 2.0], [0.5, -3.0]], dtype=np.float32)
    out = relu.forward(x, training=True)
    np.testing.assert_array_equal(out, np.array([[0.0, 2.0], [0.5, 0.0]], dtype=np.float32))

    dL_dout = np.ones_like(x)
    dL_dx = relu.backward(dL_dout)
    np.testing.assert_array_equal(dL_dx, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))
