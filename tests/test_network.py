"""Integration tests for Sequential network training and evaluation."""

import numpy as np
from src.cnn.network import Sequential
from src.cnn.layers.conv import Conv2D
from src.cnn.layers.pool import MaxPool2D
from src.cnn.layers.flatten import Flatten
from src.cnn.layers.dense import Dense
from src.cnn.layers.activations import ReLU


def test_sequential_end_to_end():
    model = Sequential([
        Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Flatten(),
        Dense(in_features=4 * 14 * 14, out_features=10),
    ])

    X_dummy = np.random.randn(10, 1, 28, 28).astype(np.float32)
    y_dummy = np.random.randint(0, 10, size=(10,))

    history = model.fit(X_dummy, y_dummy, epochs=2, batch_size=5, verbose=False)
    assert len(history["loss"]) == 2

    loss, acc = model.evaluate(X_dummy, y_dummy)
    assert isinstance(loss, float)
    assert 0.0 <= acc <= 100.0
