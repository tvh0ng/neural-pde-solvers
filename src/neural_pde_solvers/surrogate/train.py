"""End-to-end training pipeline for surrogate models."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import keras
import numpy as np
import tensorflow as tf

from neural_pde_solvers.surrogate.config import ExperimentConfig
from neural_pde_solvers.surrogate.data import DatasetSplits, load_splits
from neural_pde_solvers.surrogate.metrics import R2Score, regression_report
from neural_pde_solvers.surrogate.models import build_model, reshape_inputs
from neural_pde_solvers.utils import get_logger, save_yaml, set_seed

log = get_logger("surrogate")


def configure_precision(setting: bool | str) -> str:
    """Enable ``mixed_float16`` when requested (``"auto"`` = only if a GPU is visible)."""
    has_gpu = bool(tf.config.list_physical_devices("GPU"))
    use = has_gpu if setting == "auto" else bool(setting)
    policy = "mixed_float16" if use else "float32"
    keras.mixed_precision.set_global_policy(policy)
    return policy


def make_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


@dataclass
class SurrogateResult:
    model: keras.Model
    history: dict[str, list[float]]
    metrics: dict[str, Any]
    splits: DatasetSplits
    layout: str
    output_dir: Path

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        """Predict in physical units from raw (unscaled) inputs of shape ``(N, n_in)``."""
        xs = self.splits.scaler_x.transform(x_raw).astype(np.float32)
        ys = self.model.predict(reshape_inputs(xs, self.layout), verbose=0)
        return self.splits.scaler_y.inverse_transform(ys)


def run_experiment(cfg: ExperimentConfig, splits: DatasetSplits | None = None) -> SurrogateResult:
    """Train, evaluate and save a surrogate model described by ``cfg``.

    Artifacts written to ``cfg.output_dir``: ``best.keras`` (lowest validation loss),
    ``history.csv``, ``metrics.json``, ``scalers.joblib``, ``config.yaml``.
    """
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    tc = cfg.training
    policy = configure_precision(tc.mixed_precision)
    log.info("Precision policy: %s", policy)

    splits = splits or load_splits(cfg.data, cfg.seed)
    log.info("Data: %s", splits.summary())

    model, layout = build_model(
        cfg.model, cfg.data.n_inputs, cfg.data.n_outputs, **cfg.model_params
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=tc.learning_rate),
        loss="mse",
        metrics=[R2Score(), keras.metrics.MeanAbsoluteError(name="mae")],
    )
    log.info("Model %s: %s parameters", model.name, f"{model.count_params():,}")

    x_tr = reshape_inputs(splits.x_train, layout)
    x_te = reshape_inputs(splits.x_test, layout)
    train_ds = make_dataset(x_tr, splits.y_train, tc.batch_size, True, cfg.seed)
    has_val = splits.x_val is not None
    if has_val:
        x_va = reshape_inputs(splits.x_val, layout)
        val_ds = make_dataset(x_va, splits.y_val, tc.batch_size, False, cfg.seed)
        monitor = "val_loss"
    else:
        log.warning("val_ratio = 0: early stopping monitors the training loss.")
        val_ds, monitor = None, "loss"

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=monitor, patience=tc.patience_es, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=tc.lr_factor,
            patience=tc.patience_lr,
            min_lr=tc.min_lr,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(out / "best.keras", monitor=monitor, save_best_only=True),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]

    t0 = time.perf_counter()
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=tc.epochs,
        callbacks=callbacks,
        verbose=tc.verbose,
        shuffle=False,  # tf.data pipeline already shuffles
    )
    train_time = time.perf_counter() - t0

    # ---------------------------------------------------------------- evaluation
    def report(x, y):
        pred = model.predict(x, batch_size=tc.batch_size, verbose=0)
        return {
            "scaled": regression_report(y, pred),
            "physical": regression_report(
                splits.scaler_y.inverse_transform(y), splits.scaler_y.inverse_transform(pred)
            ),
        }

    metrics: dict[str, Any] = {"train": report(x_tr, splits.y_train)}
    if has_val:
        metrics["val"] = report(x_va, splits.y_val)
    metrics["test"] = report(x_te, splits.y_test)
    metrics.update(
        train_time_s=train_time,
        epochs_run=len(hist.history["loss"]),
        n_parameters=int(model.count_params()),
        precision_policy=policy,
    )
    log.info(
        "Test (physical units): RMSE %.4g | MAE %.4g | R² %.5f",
        metrics["test"]["physical"]["rmse"],
        metrics["test"]["physical"]["mae"],
        metrics["test"]["physical"]["r2"],
    )

    # ---------------------------------------------------------------- artifacts
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump({"x": splits.scaler_x, "y": splits.scaler_y}, out / "scalers.joblib")
    save_yaml(cfg.to_dict(), out / "config.yaml")
    log.info("Artifacts written to %s", out)

    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    return SurrogateResult(model, history, metrics, splits, layout, out)


def load_trained(output_dir: str | Path) -> tuple[keras.Model, dict, ExperimentConfig]:
    """Reload ``best.keras``, the scalers and the config saved by :func:`run_experiment`."""
    out = Path(output_dir)
    model = keras.models.load_model(out / "best.keras")
    scalers = joblib.load(out / "scalers.joblib")
    return model, scalers, ExperimentConfig.from_yaml(out / "config.yaml")
