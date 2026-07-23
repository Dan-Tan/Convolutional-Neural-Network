"""Utility functions for array manipulation, dataset loading, and encoding."""

import os
import gzip
import urllib.request
import numpy as np
from typing import Tuple, Optional


def get_im2col_indices(
    x_shape: Tuple[int, int, int, int],
    field_height: int,
    field_width: int,
    padding: int = 1,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate index arrays for im2col transformation.

    Args:
        x_shape: (N, C, H, W) shape of input tensor.
        field_height: Height of convolution kernel.
        field_width: Width of convolution kernel.
        padding: Zero-padding amount.
        stride: Stride step size.

    Returns:
        Tuple of (k, i, j) index arrays for im2col slicing.
    """
    N, C, H, W = x_shape
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k.astype(int), i.astype(int), j.astype(int)


def im2col_indices(
    x: np.ndarray,
    field_height: int,
    field_width: int,
    padding: int = 1,
    stride: int = 1,
) -> np.ndarray:
    """Transform batch image tensor into column matrix for fast convolution.

    Args:
        x: Input array of shape (N, C, H, W).
        field_height: Kernel height.
        field_width: Kernel width.
        padding: Padding size.
        stride: Stride size.

    Returns:
        Column matrix of shape (C * field_height * field_width, N * out_height * out_width).
    """
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant")
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(
    cols: np.ndarray,
    x_shape: Tuple[int, int, int, int],
    field_height: int = 3,
    field_width: int = 3,
    padding: int = 1,
    stride: int = 1,
) -> np.ndarray:
    """Transform column matrix back into batch image tensor gradient.

    Args:
        cols: Matrix of gradients wrt columns.
        x_shape: Original input shape (N, C, H, W).
        field_height: Kernel height.
        field_width: Kernel width.
        padding: Padding size.
        stride: Stride size.

    Returns:
        Gradient tensor of shape (N, C, H, W).
    """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)

    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer class indices to one-hot encoded matrix.

    Args:
        y: 1D array of class labels.
        num_classes: Total number of classes.

    Returns:
        2D array of shape (N, num_classes).
    """
    encoded = np.zeros((y.size, num_classes), dtype=np.float32)
    encoded[np.arange(y.size), y] = 1.0
    return encoded


def load_mnist(
    data_dir: str = "./data",
    num_train: int = 10000,
    num_test: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset with automatic download fallback and synthetic data fallback.

    Args:
        data_dir: Directory to save downloaded files.
        num_train: Number of training samples to load.
        num_test: Number of test samples to load.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test).
    """
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_lbl": "t10k-labels-idx1-ubyte.gz",
    }

    paths = {}
    for key, fname in files.items():
        fpath = os.path.join(data_dir, fname)
        paths[key] = fpath
        if not os.path.exists(fpath):
            try:
                url = base_url + fname
                urllib.request.urlretrieve(url, fpath)
            except Exception:
                pass

    try:
        with gzip.open(paths["train_img"], "rb") as f:
            X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
        with gzip.open(paths["train_lbl"], "rb") as f:
            y_train = np.frombuffer(f.read(), np.uint8, offset=8)
        with gzip.open(paths["test_img"], "rb") as f:
            X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
        with gzip.open(paths["test_lbl"], "rb") as f:
            y_test = np.frombuffer(f.read(), np.uint8, offset=8)

        X_train = X_train[:num_train].astype(np.float32) / 255.0
        y_train = y_train[:num_train]
        X_test = X_test[:num_test].astype(np.float32) / 255.0
        y_test = y_test[:num_test]

        return X_train, y_train, X_test, y_test
    except Exception:
        # Fallback to deterministic synthetic dataset for offline environments / quick tests
        np.random.seed(42)
        X_train = np.random.randn(num_train, 1, 28, 28).astype(np.float32)
        y_train = np.random.randint(0, 10, size=(num_train,))
        X_test = np.random.randn(num_test, 1, 28, 28).astype(np.float32)
        y_test = np.random.randint(0, 10, size=(num_test,))
        return X_train, y_train, X_test, y_test
