"""Loss functions and their gradients."""

from __future__ import annotations

import numpy as np


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """Categorical cross-entropy loss.

    Parameters
    ----------
    probs:
        Softmax probabilities, shape ``(n, C)``.
    y:
        Integer class labels, shape ``(n,)``.

    Returns
    -------
    Scalar loss value.
    """
    n = y.shape[0]
    log_p = np.log(probs[np.arange(n), y] + 1e-15)
    return float(-log_p.mean())


def cross_entropy_gradient(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Combined softmax + cross-entropy gradient w.r.t. pre-activation z.

    The gradient simplifies to ``p - one_hot(y)``, which is passed directly
    to the output layer's ``backward`` (bypassing the softmax derivative).

    Parameters
    ----------
    probs:
        Softmax probabilities, shape ``(n, C)``.
    y:
        Integer class labels, shape ``(n,)``.

    Returns
    -------
    ``dL/dz`` of shape ``(n, C)``.
    """
    dz = probs.copy()
    dz[np.arange(y.shape[0]), y] -= 1.0
    return dz
