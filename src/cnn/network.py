"""Sequential Neural Network container."""

import numpy as np
from typing import List, Tuple, Optional, Dict
from src.cnn.layers.base import Layer
from src.cnn.layers.activations import SoftmaxCrossEntropyLoss
from src.cnn.utils import one_hot_encode


class Sequential:
    """Sequential Convolutional Neural Network container."""

    def __init__(self, layers: Optional[List[Layer]] = None) -> None:
        """Initialize Sequential model container.

        Args:
            layers: Optional list of Layer instances.
        """
        self.layers: List[Layer] = layers if layers is not None else []
        self.loss_fn = SoftmaxCrossEntropyLoss()

    def add(self, layer: Layer) -> None:
        """Append a layer to the network architecture."""
        self.layers.append(layer)

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """Propagate input forward through all network layers.

        Args:
            X: Input dataset / tensor.
            training: Whether network is in training mode.

        Returns:
            Output logits from the final layer.
        """
        out = X
        for layer in self.layers:
            out = layer.forward(out, training=training)
        return out

    def backward(self, dL_dout: np.ndarray) -> None:
        """Propagate gradient backwards through all layers in reverse order."""
        grad = dL_dout
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices for input samples.

        Args:
            X: Input dataset tensor.

        Returns:
            Predicted class indices (1D array).
        """
        logits = self.forward(X, training=False)
        return np.argmax(logits, axis=-1)

    def set_learning_rate(self, lr: float) -> None:
        """Set learning rate across all trainable layers."""
        for layer in self.layers:
            if hasattr(layer, "learning_rate"):
                layer.learning_rate = lr

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evaluate network loss and accuracy on evaluation dataset.

        Args:
            X: Input images.
            y: Ground truth labels (integer indices or one-hot).

        Returns:
            Tuple of (loss, accuracy_percentage).
        """
        logits = self.forward(X, training=False)
        y_one_hot = y if y.ndim == 2 else one_hot_encode(y, num_classes=logits.shape[-1])
        y_true = np.argmax(y_one_hot, axis=-1)

        loss = self.loss_fn.forward(logits, y_one_hot)
        preds = np.argmax(logits, axis=-1)
        acc = float(np.mean(preds == y_true) * 100.0)

        return loss, acc

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 5,
        batch_size: int = 64,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train neural network using mini-batch stochastic gradient descent.

        Args:
            X_train: Training input images (N, C, H, W).
            y_train: Training class labels (integer indices or one-hot).
            epochs: Total training epochs.
            batch_size: Mini-batch size.
            X_val: Optional validation input images.
            y_val: Optional validation class labels.
            verbose: Print epoch progress metrics.

        Returns:
            History dictionary containing loss and accuracy records.
        """
        history: Dict[str, List[float]] = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        num_samples = X_train.shape[0]
        y_train_one_hot = y_train if y_train.ndim == 2 else one_hot_encode(y_train)

        for epoch in range(1, epochs + 1):
            # Shuffle dataset every epoch
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_one_hot[indices]

            epoch_loss = 0.0
            correct_preds = 0

            num_batches = int(np.ceil(num_samples / batch_size))

            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Forward pass
                logits = self.forward(X_batch, training=True)
                loss = self.loss_fn.forward(logits, y_batch)
                epoch_loss += loss * (end_idx - start_idx)

                # Accuracy accumulation
                preds = np.argmax(logits, axis=-1)
                true_labels = np.argmax(y_batch, axis=-1)
                correct_preds += int(np.sum(preds == true_labels))

                # Backward pass
                loss_grad = self.loss_fn.backward()
                self.backward(loss_grad)

            avg_loss = epoch_loss / num_samples
            train_acc = (correct_preds / num_samples) * 100.0

            history["loss"].append(avg_loss)
            history["accuracy"].append(train_acc)

            val_str = ""
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                val_str = f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}%"

            if verbose:
                print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f} - acc: {train_acc:.2f}%{val_str}")

        return history
