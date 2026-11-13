"""Matplotlib helpers for surrogate-model training and predictions."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history: dict[str, list[float]]):
    """Loss, R² and learning-rate curves from a Keras ``History.history`` dict."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    axes[0].semilogy(history["loss"], label="train")
    if "val_loss" in history:
        axes[0].semilogy(history["val_loss"], label="validation")
    axes[0].set(title="Loss (MSE, scaled)", xlabel="Epoch")
    axes[0].legend()

    if "r2" in history:
        axes[1].plot(history["r2"], label="train")
        if "val_r2" in history:
            axes[1].plot(history["val_r2"], label="validation")
        axes[1].set(title="R² (pooled)", xlabel="Epoch")
        axes[1].legend()

    lr_key = next((k for k in ("learning_rate", "lr") if k in history), None)
    if lr_key:
        axes[2].semilogy(history[lr_key])
        axes[2].set(title="Learning rate", xlabel="Epoch")
    else:
        axes[2].set_axis_off()
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sample_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, n: int = 5, seed: int = 0, ylabel: str = "Output"
):
    """Overlay ground truth and prediction for ``n`` randomly chosen samples."""
    idx = np.random.default_rng(seed).choice(len(y_true), size=min(n, len(y_true)), replace=False)
    fig, axes = plt.subplots(len(idx), 1, figsize=(12, 2.6 * len(idx)), sharex=True, squeeze=False)
    for ax, i in zip(axes[:, 0], idx, strict=True):
        ax.plot(y_true[i], label="ground truth", lw=1.8)
        ax.plot(y_pred[i], "--", label="prediction", lw=1.4)
        ax.set(title=f"Test sample #{i}", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("Output index")
    fig.tight_layout()
    return fig
