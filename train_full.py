#!/usr/bin/env python3
"""Extended Training & High-Quality Plot Generator for CNN from Scratch."""

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
    parser = argparse.ArgumentParser(description="Full training pipeline with publication-quality plots")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.015, help="Initial learning rate (default: 0.015)")
    parser.add_argument("--lr-decay", type=float, default=0.95, help="Learning rate decay per epoch")
    parser.add_argument("--num-train", type=int, default=10000, help="Number of training samples (default: 10000)")
    parser.add_argument("--num-test", type=int, default=2000, help="Number of test samples (default: 2000)")
    return parser.parse_args()


def generate_publication_plots(history, model, X_test, y_test):
    """Generate high-resolution, beautifully styled plots for documentation."""
    os.makedirs("assets", exist_ok=True)
    epochs = len(history["loss"])
    epoch_axis = range(1, epochs + 1)

    # 1. Training Curves Plot (Loss & Accuracy)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Loss panel
    ax1.plot(epoch_axis, history["loss"], label="Train Loss", color="#2563eb", linewidth=2.5, marker="o", markersize=6)
    if history["val_loss"]:
        ax1.plot(epoch_axis, history["val_loss"], label="Validation Loss", color="#dc2626", linewidth=2.2, linestyle="--", marker="s", markersize=5)
    ax1.set_title("Cross-Entropy Loss Convergence", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="medium", color="#334155")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="medium", color="#334155")
    ax1.set_xticks(epoch_axis)
    ax1.grid(True, linestyle=":", alpha=0.6, color="#94a3b8")
    ax1.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10)
    ax1.set_facecolor("#f8fafc")

    # Accuracy panel
    ax2.plot(epoch_axis, history["accuracy"], label="Train Accuracy", color="#2563eb", linewidth=2.5, marker="o", markersize=6)
    if history["val_accuracy"]:
        ax2.plot(epoch_axis, history["val_accuracy"], label="Validation Accuracy", color="#16a34a", linewidth=2.2, linestyle="--", marker="^", markersize=6)
    ax2.set_title("Classification Accuracy (%)", fontsize=13, fontweight="bold", pad=12, color="#0f172a")
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="medium", color="#334155")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="medium", color="#334155")
    ax2.set_xticks(epoch_axis)
    ax2.grid(True, linestyle=":", alpha=0.6, color="#94a3b8")
    ax2.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=10)
    ax2.set_facecolor("#f8fafc")

    # Annotate peak accuracy
    if history["val_accuracy"]:
        best_epoch = int(np.argmax(history["val_accuracy"])) + 1
        best_acc = history["val_accuracy"][best_epoch - 1]
        ax2.annotate(
            f"Peak: {best_acc:.2f}% (Epoch {best_epoch})",
            xy=(best_epoch, best_acc),
            xytext=(best_epoch - 2.5, best_acc - 8 if best_acc > 50 else best_acc + 5),
            arrowprops=dict(facecolor="#15803d", shrink=0.08, width=1.5, headwidth=6),
            fontsize=9.5,
            fontweight="bold",
            color="#15803d",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0fdf4", edgecolor="#86efac", alpha=0.9),
        )

    plt.tight_layout()
    plot_path = os.path.join("assets", "training_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved publication-quality training curve plot to: {plot_path}")

    # 2. Extract Feature Maps
    # Find layer instances
    conv1 = model.layers[0]
    relu1 = model.layers[1]
    pool1 = model.layers[2]
    conv2 = model.layers[3]
    relu2 = model.layers[4]

    sample_img = X_test[0:1]  # shape (1, 1, 28, 28)

    # Forward pass outputs
    fm1 = conv1.forward(sample_img, training=False)
    act1 = relu1.forward(fm1, training=False)
    p1 = pool1.forward(act1, training=False)
    fm2 = conv2.forward(p1, training=False)
    act2 = relu2.forward(fm2, training=False)

    # Layer 1 Feature Maps (16 subplots)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9), dpi=300)
    fig.suptitle("Conv2D Layer 1 Feature Maps (16 Filters)", fontsize=14, fontweight="bold", y=0.95, color="#0f172a")
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(act1[0, i, :, :], cmap="viridis", interpolation="nearest")
        ax.set_title(f"Filter {i+1}", fontsize=9, color="#334155", fontweight="medium")
        ax.axis("off")
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    fm1_path = os.path.join("assets", "feature_maps_layer1.png")
    plt.savefig(fm1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Layer 1 feature maps to: {fm1_path}")

    # Layer 2 Feature Maps (16 subplots selected from 16 channels)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9), dpi=300)
    fig.suptitle("Conv2D Layer 2 Feature Maps (16 Filters)", fontsize=14, fontweight="bold", y=0.95, color="#0f172a")
    num_filters_l2 = min(16, act2.shape[1])
    for i, ax in enumerate(axes.flat):
        if i < num_filters_l2:
            ax.imshow(act2[0, i, :, :], cmap="magma", interpolation="nearest")
            ax.set_title(f"Filter {i+1}", fontsize=9, color="#334155", fontweight="medium")
        ax.axis("off")
    plt.subplots_adjust(wspace=0.15, hspace=0.25)
    fm2_path = os.path.join("assets", "feature_maps_layer2.png")
    plt.savefig(fm2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Layer 2 feature maps to: {fm2_path}")


def main():
    args = parse_args()

    print("==================================================")
    print(" CNN Extended Training & Plot Generator ")
    print("==================================================")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Initial LR: {args.lr}")
    print(f"Training Samples: {args.num_train} | Test Samples: {args.num_test}")
    print("--------------------------------------------------")

    print("Loading MNIST dataset...")
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

    print("\nStarting training process...")
    current_lr = args.lr
    history = {
        "loss": [],
        "accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} (Learning Rate: {current_lr:.5f}) ---")
        model.set_learning_rate(current_lr)

        ep_hist = model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=args.batch_size,
            X_val=X_test,
            y_val=y_test,
            verbose=True,
        )

        history["loss"].append(ep_hist["loss"][0])
        history["accuracy"].append(ep_hist["accuracy"][0])
        history["val_loss"].append(ep_hist["val_loss"][0])
        history["val_accuracy"].append(ep_hist["val_accuracy"][0])

        # Apply learning rate decay
        current_lr *= args.lr_decay

    final_loss, final_acc = model.evaluate(X_test, y_test)
    print("--------------------------------------------------")
    print(f"Final Evaluation -> Loss: {final_loss:.4f} | Accuracy: {final_acc:.2f}%")
    print("--------------------------------------------------")

    print("\nGenerating publication-quality plots...")
    generate_publication_plots(history, model, X_test, y_test)
    print("\nTraining and plot generation complete!")


if __name__ == "__main__":
    main()
