"""Regression metrics (Keras streaming metric + NumPy report)."""

from __future__ import annotations

import keras
import numpy as np
from keras import ops
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@keras.saving.register_keras_serializable(package="neural_pde_solvers")
class R2Score(keras.metrics.Metric):
    """Coefficient of determination accumulated over *all* batches of an epoch.

    Computing ``1 - SS_res / SS_tot`` per batch and averaging is biased because each
    batch uses its own mean. Here ``SS_res``, ``sum(y)``, ``sum(y^2)`` and ``n`` are
    accumulated, and ``SS_tot = sum(y^2) - n * mean(y)^2`` is formed at the end.

    All output columns are pooled (variance-weighted R²). For the per-output average
    used by ``sklearn.metrics.r2_score`` see :func:`regression_report`.
    """

    def __init__(self, name: str = "r2", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(shape=(), initializer="zeros", name="ss_res")
        self.y_sum = self.add_weight(shape=(), initializer="zeros", name="y_sum")
        self.y_sq_sum = self.add_weight(shape=(), initializer="zeros", name="y_sq_sum")
        self.count = self.add_weight(shape=(), initializer="zeros", name="count")

    def update_state(self, y_true, y_pred, sample_weight=None):  # noqa: ARG002
        y_true = ops.cast(y_true, "float32")
        y_pred = ops.cast(y_pred, "float32")
        self.ss_res.assign_add(ops.sum(ops.square(y_true - y_pred)))
        self.y_sum.assign_add(ops.sum(y_true))
        self.y_sq_sum.assign_add(ops.sum(ops.square(y_true)))
        self.count.assign_add(ops.cast(ops.size(y_true), "float32"))

    def result(self):
        eps = keras.backend.epsilon()
        mean = self.y_sum / (self.count + eps)
        ss_tot = self.y_sq_sum - self.count * ops.square(mean)
        return 1.0 - self.ss_res / (ss_tot + eps)

    def reset_state(self):
        for v in self.variables:
            v.assign(ops.zeros_like(v))


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MSE, RMSE, MAE and R² (uniform average over output columns)."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
