"""Loading, splitting and scaling tabular data for surrogate models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from neural_pde_solvers.surrogate.config import DataConfig


@dataclass
class DatasetSplits:
    """Scaled 2-D arrays plus the fitted scalers (to map predictions back to physical units)."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray | None
    y_val: np.ndarray | None
    x_test: np.ndarray
    y_test: np.ndarray
    scaler_x: MinMaxScaler | StandardScaler
    scaler_y: MinMaxScaler | StandardScaler

    def summary(self) -> str:
        val = "none" if self.x_val is None else f"{self.x_val.shape}"
        return (
            f"train {self.x_train.shape} -> {self.y_train.shape} | val {val} | "
            f"test {self.x_test.shape} -> {self.y_test.shape}"
        )


def load_table(path: str, n_samples: int | None = None) -> np.ndarray:
    """Read a header-less numeric CSV into a float array (first ``n_samples`` rows)."""
    df = pd.read_csv(path, header=None, nrows=n_samples)
    return df.to_numpy(dtype=np.float64)


def _make_scaler(kind: str):
    return MinMaxScaler() if kind == "minmax" else StandardScaler()


def split_and_scale(table: np.ndarray, cfg: DataConfig, seed: int = 42) -> DatasetSplits:
    """Split rows into train/val/test and scale with statistics from *train only*."""
    n_rows, n_cols = table.shape
    stop = max(cfg.input_cols[1], cfg.output_cols[1])
    if stop > n_cols:
        raise ValueError(f"Config needs {stop} columns but the table has only {n_cols}.")
    if cfg.n_samples is not None and n_rows < cfg.n_samples:
        raise ValueError(f"Requested n_samples={cfg.n_samples} but the table has {n_rows} rows.")

    X = table[:, slice(*cfg.input_cols)]
    Y = table[:, slice(*cfg.output_cols)]
    idx = np.arange(n_rows)
    if cfg.shuffle:
        np.random.default_rng(seed).shuffle(idx)

    n_trainval = int(n_rows * (1.0 - cfg.test_ratio))
    n_val = int(n_trainval * cfg.val_ratio)
    n_train = n_trainval - n_val
    if n_train < 1 or n_rows - n_trainval < 1:
        raise ValueError("Not enough rows for the requested split.")
    i_train, i_val, i_test = idx[:n_train], idx[n_train:n_trainval], idx[n_trainval:]

    sx, sy = _make_scaler(cfg.scaler), _make_scaler(cfg.scaler)
    sx.fit(X[i_train])
    sy.fit(Y[i_train])

    def tx(i):
        return sx.transform(X[i]).astype(np.float32)

    def ty(i):
        return sy.transform(Y[i]).astype(np.float32)

    return DatasetSplits(
        x_train=tx(i_train),
        y_train=ty(i_train),
        x_val=tx(i_val) if n_val else None,
        y_val=ty(i_val) if n_val else None,
        x_test=tx(i_test),
        y_test=ty(i_test),
        scaler_x=sx,
        scaler_y=sy,
    )


def load_splits(cfg: DataConfig, seed: int = 42) -> DatasetSplits:
    return split_and_scale(load_table(cfg.path, cfg.n_samples), cfg, seed)
