"""Multi-layer perceptron for binary breast-cancer classification.

Fixed architecture
------------------
  Input  : 30 features
  Hidden1: 16 units, ReLU, He init
  Hidden2:  8 units, ReLU, He init
  Output :  2 units, Softmax, He init
"""

from __future__ import annotations

import numpy as np

from src.layers import DenseLayer
from src.losses import cross_entropy, cross_entropy_gradient
from src.metrics import accuracy

# (n_inputs, n_units, activation)
_DEFAULT_ARCH: list[tuple[int, int, str]] = [
    (30, 24, "relu"),
    (24, 24, "relu"),
    (24, 2, "softmax"),
]


class MLP:
    """Vanilla mini-batch-free MLP trained with full-batch gradient descent.

    Parameters
    ----------
    seed:
        Optional seed forwarded to each layer for reproducible initialisation.
    arch:
        List of ``(n_inputs, n_units, activation)`` tuples.  Defaults to the
        project-specified architecture.
    """

    def __init__(
        self,
        seed: int | None = None,
        arch: list[tuple[int, int, str]] | None = None,
    ) -> None:
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
        verbose: bool = True,
        log_every: int = 100,
    ) -> dict[str, list[float]]:
        """Train the network with full-batch gradient descent.

        Parameters
        ----------
        X_train, y_train:
            Training features (n, 30) and integer labels (n,).
        X_val, y_val:
            Optional validation set; loss is tracked but NOT used for updates.
        lr:
            Learning rate.
        epochs:
            Total number of gradient steps.
        verbose:
            Print loss every ``log_every`` epochs when ``True``.
        log_every:
            Print interval (default 100).

        Returns
        -------
        ``history`` dict with keys ``"train_loss"`` and (if a validation
        set was provided) ``"val_loss"``.
        """
        history: dict[str, list[float]] = {"train_loss": [], "train_acc": []}
        if X_val is not None:
            history["val_loss"] = []
            history["val_acc"] = []

        completed = 0
        try:
            for epoch in range(1, epochs + 1):
                # --- forward + backward + update ---
                probs = self.forward(X_train)
                train_loss = cross_entropy(probs, y_train)
                self.backward(y_train)
                self.update(lr)

                # --- metrics (2 extra forward passes per epoch) ---
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

        return history
