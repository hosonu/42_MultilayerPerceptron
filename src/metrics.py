"""Evaluation metrics and learning-curve visualisation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples."""
    return float((y_true == y_pred).mean())


def plot_learning_curves(
    history: dict[str, list[float]],
    *,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot loss and accuracy learning curves side by side.

    Parameters
    ----------
    history:
        Dict returned by :meth:`MLP.fit`.  Expected keys:
        ``"train_loss"``, ``"val_loss"`` (optional),
        ``"train_acc"``,  ``"val_acc"``  (optional).
    save_path:
        If given, save the figure to this path before displaying.
    show:
        Call ``plt.show()`` when ``True`` (default).
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    has_val = "val_loss" in history and len(history["val_loss"]) > 0

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Learning Curves", fontsize=14, fontweight="bold")

    # --- Loss ---
    ax_loss.plot(epochs, history["train_loss"], label="train", linewidth=1.8)
    if has_val:
        ax_loss.plot(
            epochs,
            history["val_loss"],
            label="validation",
            linestyle="--",
            linewidth=1.8,
        )
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-Entropy Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax_acc.plot(epochs, history["train_acc"], label="train", linewidth=1.8)
    if has_val and "val_acc" in history:
        ax_acc.plot(
            epochs,
            history["val_acc"],
            label="validation",
            linestyle="--",
            linewidth=1.8,
        )
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Learning curves saved to {save_path}")

    if show:
        plt.show()

    plt.close(fig)
