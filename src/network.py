"""Multi-layer perceptron for binary breast-cancer classification.

Fixed architecture
------------------
  Input  : 30 features
  Hidden1: 32 units, ReLU, He init
  Hidden2: 32 units, ReLU, He init
  Output :  2 units, Softmax, He init
"""

from __future__ import annotations

import copy
import numpy as np

from src.layers import DenseLayer
from src.losses import cross_entropy, cross_entropy_gradient
from src.metrics import accuracy

# (n_inputs, n_units, activation)
_DEFAULT_ARCH: list[tuple[int, int, str]] = [
    (30, 32, "relu"),
    (32, 32, "relu"),
    (32, 2, "softmax"),
]


class MLP:
    """MLP trained with mini-batch SGD.

    Parameters
    ----------
    seed:
        Optional seed forwarded to each layer for reproducible initialisation
        and used as the default shuffle seed in :meth:`fit`.
    arch:
        List of ``(n_inputs, n_units, activation)`` tuples.  Defaults to the
        project-specified architecture.
    """

    def __init__(
        self,
        seed: int | None = None,
        arch: list[tuple[int, int, str]] | None = None,
    ) -> None:
        self.seed = seed
        spec = arch if arch is not None else _DEFAULT_ARCH
        layer_seed = seed
        self.layers: list[DenseLayer] = []
        for n_in, n_out, act in spec:
            self.layers.append(DenseLayer(n_in, n_out, act, seed=layer_seed))
            if layer_seed is not None:
                layer_seed += 1

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run a forward pass and return softmax probabilities (n, 2)."""
        out: np.ndarray = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices; 0 = Benign, 1 = Malignant."""
        return self.forward(X).argmax(axis=1)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Cross-entropy loss on (X, y); triggers a fresh forward pass."""
        return cross_entropy(self.forward(X), y)

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backward(self, y: np.ndarray) -> None:
        """Backpropagate cross-entropy loss using cached forward values.

        Must be called immediately after :meth:`forward` so that each
        layer's internal cache (``_output``, ``_z``) is up-to-date.

        The combined softmax + cross-entropy gradient simplifies to
        ``(p - one_hot(y)) / n`` at the pre-activation level (dL/dz),
        which is passed directly to the output layer's ``backward``.
        """
        p: np.ndarray = self.layers[-1]._output  # type: ignore[assignment]
        grad = cross_entropy_gradient(p, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr: float) -> None:
        """Apply a vanilla SGD step to every layer."""
        for layer in self.layers:
            layer.update(lr)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        lr: float = 0.01,
        epochs: int = 1_000,
        batch_size: int = 32,
        seed: int | None = None,
        verbose: bool = True,
        log_every: int = 100,
        patience: int = 20,
        min_delta: float = 1e-4,
    ) -> dict[str, list[float]]:
        """Train the network with mini-batch SGD and early stopping.

        Parameters
        ----------
        X_train, y_train:
            Training features (n, 30) and integer labels (n,).
        X_val, y_val:
            Optional validation set; loss is tracked but NOT used for updates.
        lr:
            Learning rate.
        epochs:
            Maximum number of passes over the training set.
        batch_size:
            Mini-batch size.  If ``>= n_train``, falls back to full-batch.
        seed:
            RNG seed for per-epoch shuffling.  Defaults to ``self.seed``.
        verbose:
            Print loss every ``log_every`` epochs when ``True``.
        log_every:
            Print interval (default 100).
        patience:
            Early stopping: stop after this many epochs with no improvement
            in val_loss.  Ignored when no validation set is provided.
        min_delta:
            Minimum improvement in val_loss to count as a new best.

        Returns
        -------
        ``history`` dict with keys ``"train_loss"`` / ``"train_acc"`` and
        (if a validation set was provided) ``"val_loss"`` / ``"val_acc"``.
        Metrics are computed on the full sets after each epoch.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        history: dict[str, list[float]] = {"train_loss": [], "train_acc": []}
        if X_val is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        n_train = X_train.shape[0]
        effective_bs = min(batch_size, n_train)
        shuffle_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(shuffle_seed)

        # --- early stopping state ---
        best_val_loss: float = float("inf")
        patience_counter: int = 0
        best_weights: list[dict[str, np.ndarray]] | None = None

        completed = 0
        try:
            for epoch in range(1, epochs + 1):
                # --- mini-batch SGD ---
                indices = rng.permutation(n_train)
                for start in range(0, n_train, effective_bs):
                    batch_idx = indices[start : start + effective_bs]
                    X_batch = X_train[batch_idx]
                    y_batch = y_train[batch_idx]
                    self.forward(X_batch)
                    self.backward(y_batch)
                    self.update(lr)

                # --- metrics on full sets (after all batches) ---
                train_loss = self.loss(X_train, y_train)
                train_acc = accuracy(y_train, self.predict(X_train))
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)

                val_loss: float | None = None
                val_acc: float | None = None
                if X_val is not None and y_val is not None:
                    val_loss = self.loss(X_val, y_val)
                    val_acc = accuracy(y_val, self.predict(X_val))
                    history["val_loss"].append(val_loss)
                    history["val_acc"].append(val_acc)

                    # --- early stopping check ---
                    if val_loss < best_val_loss - min_delta:
                        best_val_loss = val_loss
                        best_weights = [
                            {"W": copy.deepcopy(l.W), "b": copy.deepcopy(l.b)}
                            for l in self.layers
                        ]
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(
                                    f"\nEarly stopping at epoch {epoch}"
                                    f" (best val_loss={best_val_loss:.4f})"
                                )
                            break

                if verbose and (epoch == 1 or epoch % log_every == 0):
                    msg = (
                        f"epoch {epoch:>5}/{epochs}"
                        f"  train_loss={train_loss:.4f}"
                        f"  train_acc={train_acc:.4f}"
                    )
                    if val_loss is not None and val_acc is not None:
                        msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                    print(msg)

                completed = epoch

        except KeyboardInterrupt:
            print(
                f"\nTraining interrupted after epoch {completed}/{epochs}."
                " Returning partial history."
            )

        # --- restore best weights ---
        if best_weights is not None:
            for layer, w in zip(self.layers, best_weights):
                layer.W = w["W"]
                layer.b = w["b"]

        return history
