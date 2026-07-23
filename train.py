#!/usr/bin/env python3
"""Train Convolutional Neural Network on MNIST dataset."""

import os
import multiprocessing

# Automatically utilize all CPU cores for parallel NumPy matrix operations
_num_cores = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = _num_cores
os.environ["MKL_NUM_THREADS"] = _num_cores
os.environ["OPENBLAS_NUM_THREADS"] = _num_cores
os.environ["NUMEXPR_NUM_THREADS"] = _num_cores

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.cnn import Sequential, Conv2D, MaxPool2D, Flatten, Dense, ReLU, load_mnist


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN from scratch on MNIST dataset")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num-train", type=int, default=5000, help="Number of training samples")
    parser.add_argument("--num-test", type=int, default=1000, help="Number of test samples")
    parser.add_argument("--save-plot", action="store_true", default=True, help="Save training curves")
    return parser.parse_args()


def main():
    args = parse_args()

    print("==================================================")
    print(" CNN from Scratch - Training Pipeline ")
    print("==================================================")
    print(f"Hardware Threads: {multiprocessing.cpu_count()} (Multi-Threaded Parallel Execution)")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Training Samples: {args.num_train} | Test Samples: {args.num_test}")
    print("--------------------------------------------------")

    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_mnist(
        num_train=args.num_train,
        num_test=args.num_test,
    )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Build CNN Architecture
    # 28x28 -> Conv(8, 3x3, p=1) -> 28x28 -> MaxPool(2x2) -> 14x14
    # -> Conv(16, 3x3, p=1) -> 14x14 -> MaxPool(2x2) -> 7x7
    # -> Flatten (16*7*7 = 784) -> Dense(64) -> Dense(10)
    model = Sequential([
        Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=1, learning_rate=args.lr),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Conv2D(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1, learning_rate=args.lr),
        ReLU(),
        MaxPool2D(pool_size=2, stride=2),
        Flatten(),
        Dense(in_features=16 * 7 * 7, out_features=64, learning_rate=args.lr),
        ReLU(),
        Dense(in_features=64, out_features=10, learning_rate=args.lr),
    ])

    print("\nStarting training loop...")
    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_test,
        y_val=y_test,
        verbose=True,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nFinal Test Evaluation - Loss: {test_loss:.4f} - Accuracy: {test_acc:.2f}%")

    if args.save_plot:
        os.makedirs("assets", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        ax1.plot(range(1, args.epochs + 1), history["loss"], label="Train Loss", marker="o", color="#2563eb")
        if history["val_loss"]:
            ax1.plot(range(1, args.epochs + 1), history["val_loss"], label="Val Loss", marker="s", color="#dc2626")
        ax1.set_title("Cross-Entropy Loss Curve", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend()

        ax2.plot(range(1, args.epochs + 1), history["accuracy"], label="Train Acc", marker="o", color="#2563eb")
        if history["val_accuracy"]:
            ax2.plot(range(1, args.epochs + 1), history["val_accuracy"], label="Val Acc", marker="s", color="#dc2626")
        ax2.set_title("Classification Accuracy (%)", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plot_path = os.path.join("assets", "training_curves.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Training plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
