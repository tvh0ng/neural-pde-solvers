"""Matplotlib helpers for PINN results."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_history(history: dict[str, list[float]], keys: Iterable[str] | None = None, ax=None):
    """Semilog plot of the total loss and each loss term against iteration."""
    ax = ax or plt.gca()
    epochs = history["epoch"]
    reserved = {"epoch", "lr"}
    keys = (
        list(keys)
        if keys is not None
        else [
            k
            for k in history
            if k not in reserved and (k == "total" or k in {"pde", "bc", "ic", "data"})
        ]
    )
    for k in keys:
        ax.semilogy(epochs, history[k], label=k, alpha=0.85, lw=1.2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_field(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    ax=None,
    title: str = "",
    xlabel: str = "x",
    ylabel: str = "y",
    cmap: str = "RdBu_r",
    levels: int = 50,
):
    """Filled contour of a scalar field on a meshgrid, with a colour bar."""
    ax = ax or plt.gca()
    c = ax.contourf(X, Y, U, levels=levels, cmap=cmap)
    plt.colorbar(c, ax=ax)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return ax


def plot_prediction_vs_exact_2d(
    mesh: list[np.ndarray],
    pred: np.ndarray,
    exact: np.ndarray,
    labels: tuple[str, str] = ("x", "y"),
    cmap: str = "viridis",
):
    """Three panels: prediction, reference and absolute error on a 2D grid."""
    Xm, Ym = mesh
    P, E = pred.reshape(Xm.shape), exact.reshape(Xm.shape)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    plot_field(Xm, Ym, P, axes[0], "PINN prediction", *labels, cmap=cmap)
    plot_field(Xm, Ym, E, axes[1], "Reference solution", *labels, cmap=cmap)
    err = np.abs(P - E)
    plot_field(Xm, Ym, err, axes[2], f"|error|  (max {err.max():.2e})", *labels, cmap="magma_r")
    fig.tight_layout()
    return fig, axes
