#!/usr/bin/env python3
"""Visualize Convolutional Layer feature maps."""

import os
import numpy as np
import matplotlib.pyplot as plt

from src.cnn import Sequential, Conv2D, MaxPool2D, Flatten, Dense, ReLU, load_mnist


def main():
    print("Generating Feature Map Visualizations...")
    X_train, y_train, _, _ = load_mnist(num_train=10, num_test=10)

    # Initialize model
    conv1 = Conv2D(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1)
    relu1 = ReLU()
    pool1 = MaxPool2D(pool_size=2, stride=2)
    conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1)
    relu2 = ReLU()

    sample_img = X_train[0:1]  # shape (1, 1, 28, 28)

    # Forward pass step by step to extract intermediate feature maps
    fm1 = conv1.forward(sample_img, training=False)
    act1 = relu1.forward(fm1, training=False)
    p1 = pool1.forward(act1, training=False)
    fm2 = conv2.forward(p1, training=False)
    act2 = relu2.forward(fm2, training=False)

    os.makedirs("assets", exist_ok=True)

    # Plot Input & Conv Layer 1 Feature Maps (16 filters)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Conv2D Layer 1 Feature Maps (16 Filters)", fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        ax.imshow(act1[0, i, :, :], cmap="viridis")
        ax.set_title(f"Filter {i+1}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fm1_path = os.path.join("assets", "feature_maps_layer1.png")
    plt.savefig(fm1_path, dpi=300)
    plt.close()
    print(f"Layer 1 feature maps saved to: {fm1_path}")

    # Plot Conv Layer 2 Feature Maps (32 filters - sample top 16)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle("Conv2D Layer 2 Feature Maps (Top 16 / 32 Filters)", fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        ax.imshow(act2[0, i, :, :], cmap="magma")
        ax.set_title(f"Filter {i+1}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fm2_path = os.path.join("assets", "feature_maps_layer2.png")
    plt.savefig(fm2_path, dpi=300)
    plt.close()
    print(f"Layer 2 feature maps saved to: {fm2_path}")


if __name__ == "__main__":
    main()
