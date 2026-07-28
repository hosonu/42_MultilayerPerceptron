"""Dense (fully-connected) layer with He initialisation and backpropagation."""

from __future__ import annotations

import numpy as np

from src.activations import relu, relu_derivative, softmax

_SUPPORTED_ACTIVATIONS = ("relu", "softmax")


class DenseLayer:
    """Single fully-connected layer.

    Parameters
    ----------
    n_inputs:
        Number of input features (fan-in).
    n_units:
        Number of neurons in this layer (fan-out).
    activation:
        Either ``"relu"`` or ``"softmax"``.
    seed:
        Optional RNG seed for weight initialisation.
    """

    def __init__(
        self,
        n_inputs: int,
        n_units: int,
        activation: str = "relu",
        seed: int | None = None,
    ) -> None:
        if activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {_SUPPORTED_ACTIVATIONS}, got {activation!r}"
            )
        self.n_inputs = n_inputs
        self.n_units = n_units
        self.activation_name = activation

        rng = np.random.default_rng(seed)
        he_scale = np.sqrt(2.0 / n_inputs)
        self.W: np.ndarray = rng.standard_normal((n_inputs, n_units)) * he_scale
        self.b: np.ndarray = np.zeros((1, n_units))

        # Accumulated gradients (set by backward)
        self.dW: np.ndarray = np.zeros_like(self.W)
        self.db: np.ndarray = np.zeros_like(self.b)

        # Forward-pass cache for backprop
        self._input: np.ndarray | None = None
        self._z: np.ndarray | None = None
        self._output: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute z = x @ W + b and apply activation; cache for backprop."""
        self._input = x
        self._z = x @ self.W + self.b

        if self.activation_name == "relu":
            self._output = relu(self._z)
        else:
            self._output = softmax(self._z)

        return self._output

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backpropagate gradients and accumulate dW, db.

        Parameters
        ----------
        grad:
            For ReLU layers: ``dL/da`` (gradient w.r.t. layer output).
            For the softmax output layer: ``dL/dz`` (the combined
            softmax + cross-entropy gradient, already simplified to
            ``p - y_onehot``).  The caller is responsible for passing
            the correct quantity.

        Returns
        -------
        ``dL/dx``, the gradient to propagate to the previous layer.
        """
        if self._input is None or self._z is None:
            raise RuntimeError("forward() must be called before backward().")

        if self.activation_name == "relu":
            dz = grad * relu_derivative(self._z)
        else:
            # Softmax + cross-entropy gradient is passed as dL/dz directly.
            dz = grad

        n = self._input.shape[0]
        self.dW = self._input.T @ dz / n
        self.db = dz.mean(axis=0, keepdims=True)
        return dz @ self.W.T

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def update(self, lr: float) -> None:
        """Vanilla SGD step: W -= lr * dW, b -= lr * db."""
        self.W -= lr * self.dW
        self.b -= lr * self.db
