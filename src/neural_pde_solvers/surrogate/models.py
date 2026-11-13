"""Model zoo for multi-output regression surrogates.

Every builder has the signature ``builder(n_inputs, n_outputs, **params) -> keras.Model``
and the output layer is a linear ``Dense`` in ``float32`` (required under mixed precision,
and no output activation so that the regression target is not clipped).

Input layouts
-------------
``flat``      ``(N, n_in)``      — dense networks
``sequence``  ``(N, n_in, 1)``   — each input feature is one "time step" (Conv1D, LSTM, CNN-GRU)
``step``      ``(N, 1, n_in)``   — one time step carrying all features (GRU, SimpleRNN)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import keras
import numpy as np
from keras import layers

LAYOUTS = ("flat", "sequence", "step")


def input_shape(n_inputs: int, layout: str) -> tuple[int, ...]:
    if layout == "flat":
        return (n_inputs,)
    if layout == "sequence":
        return (n_inputs, 1)
    if layout == "step":
        return (1, n_inputs)
    raise ValueError(f"layout must be one of {LAYOUTS}")


def reshape_inputs(x: np.ndarray, layout: str) -> np.ndarray:
    """Reshape a ``(N, n_in)`` array to the layout a model expects."""
    return x.reshape((x.shape[0], *input_shape(x.shape[1], layout)))


def _head(x, n_outputs: int):
    return layers.Dense(n_outputs, dtype="float32", name="output")(x)


# --------------------------------------------------------------------------- ANN
def build_ann(
    n_inputs: int,
    n_outputs: int,
    dense_units: Sequence[int] = (256, 128, 64, 32),
    activation: str = "tanh",
    kernel_initializer: str = "uniform",
    batch_norm: bool = True,
    dropout_rate: float = 0.2,
    dropout_after: Sequence[int] = (1,),
    name: str = "ANN",
) -> keras.Model:
    """Feed-forward network; ``dropout_after`` lists hidden-layer indices followed by dropout."""
    inp = keras.Input(shape=input_shape(n_inputs, "flat"), name="inputs")
    x = inp
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        if i in dropout_after and dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
    return keras.Model(inp, _head(x, n_outputs), name=name)


# --------------------------------------------------------------------------- 1D CNN
def build_cnn1d(
    n_inputs: int,
    n_outputs: int,
    conv_filters: Sequence[int] = (128, 64, 64),
    conv_kernels: Sequence[int] = (2, 2, 2),
    dense_units: Sequence[int] = (64, 32),
    dropout_rate: float = 0.2,
    pool_size: int = 2,
    name: str = "CNN1D",
) -> keras.Model:
    """Conv1D(+BN) stack -> dropout -> max-pool -> flatten -> Dense(+BN) stack."""
    if len(conv_filters) != len(conv_kernels):
        raise ValueError("conv_filters and conv_kernels must have the same length")
    length = n_inputs - sum(k - 1 for k in conv_kernels)
    if length // max(pool_size, 1) < 1:
        raise ValueError(
            f"{n_inputs} inputs are too few for kernels {list(conv_kernels)} and pool {pool_size}"
        )
    inp = keras.Input(shape=input_shape(n_inputs, "sequence"), name="inputs")
    x = inp
    for f, k in zip(conv_filters, conv_kernels, strict=True):
        x = layers.Conv1D(f, kernel_size=k, activation="relu")(x)
        x = layers.BatchNormalization()(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    if pool_size > 1:
        x = layers.MaxPooling1D(pool_size=pool_size)(x)
    x = layers.Flatten()(x)
    for units in dense_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.BatchNormalization()(x)
    return keras.Model(inp, _head(x, n_outputs), name=name)


# --------------------------------------------------------------------------- recurrent
_CELLS: dict[str, Callable[..., layers.Layer]] = {
    "rnn": lambda u, **kw: layers.SimpleRNN(u, activation="tanh", **kw),
    "gru": lambda u, **kw: layers.GRU(u, activation="tanh", recurrent_activation="sigmoid", **kw),
    "lstm": lambda u, **kw: layers.LSTM(u, activation="tanh", recurrent_activation="sigmoid", **kw),
}


def build_recurrent(
    n_inputs: int,
    n_outputs: int,
    cell: str = "gru",
    layout: str = "step",
    units: Sequence[int] = (128, 64, 32),
    conv_filters: Sequence[int] = (),
    conv_kernels: Sequence[int] = (),
    dense_units: Sequence[int] = (),
    dense_batch_norm: bool = False,
    dropout_rate: float = 0.25,
    dropout_after: int = -1,
    bidirectional: bool = False,
    name: str | None = None,
) -> keras.Model:
    """Optional Conv1D front-end -> stacked RNN/GRU/LSTM (+BN) -> Dense head.

    ``dropout_after`` is the index of the recurrent layer followed by dropout
    (``-1`` = the last one).
    """
    if cell not in _CELLS:
        raise ValueError(f"cell must be one of {sorted(_CELLS)}")
    if not units:
        raise ValueError("At least one recurrent layer is required")
    drop_idx = dropout_after % len(units)

    inp = keras.Input(shape=input_shape(n_inputs, layout), name="inputs")
    x = inp
    for f, k in zip(conv_filters, conv_kernels, strict=True):
        x = layers.Conv1D(f, kernel_size=k, activation="relu")(x)
        x = layers.BatchNormalization()(x)
    for i, u in enumerate(units):
        rnn = _CELLS[cell](u, return_sequences=i < len(units) - 1)
        x = layers.Bidirectional(rnn)(x) if bidirectional else rnn(x)
        x = layers.BatchNormalization()(x)
        if i == drop_idx and dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
    for u in dense_units:
        x = layers.Dense(u, activation="relu")(x)
        if dense_batch_norm:
            x = layers.BatchNormalization()(x)
    return keras.Model(inp, _head(x, n_outputs), name=name or cell.upper())


# --------------------------------------------------------------------------- registry
@dataclass(frozen=True)
class ModelSpec:
    builder: Callable[..., keras.Model]
    layout: str
    description: str


MODELS: dict[str, ModelSpec] = {
    "ann": ModelSpec(build_ann, "flat", "Feed-forward ANN"),
    "cnn1d": ModelSpec(build_cnn1d, "sequence", "1D convolutional network"),
    "rnn": ModelSpec(
        partial(build_recurrent, cell="rnn", layout="step", name="SimpleRNN"), "step", "SimpleRNN"
    ),
    "gru": ModelSpec(
        partial(build_recurrent, cell="gru", layout="step", name="GRU"), "step", "GRU"
    ),
    "lstm": ModelSpec(
        partial(build_recurrent, cell="lstm", layout="sequence", name="LSTM"), "sequence", "LSTM"
    ),
    "cnn_gru": ModelSpec(
        partial(
            build_recurrent, cell="gru", layout="sequence", dense_batch_norm=True, name="CNN_GRU"
        ),
        "sequence",
        "Conv1D front-end + GRU",
    ),
}


def build_model(name: str, n_inputs: int, n_outputs: int, **params) -> tuple[keras.Model, str]:
    """Build a registered model; returns ``(model, input_layout)``."""
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    spec = MODELS[name]
    params = dict(params)
    layout = params.pop("layout", spec.layout)
    if isinstance(spec.builder, partial) and spec.builder.func is build_recurrent:
        params["layout"] = layout
    elif layout != spec.layout:
        raise ValueError(f"Model '{name}' only supports the '{spec.layout}' layout")
    return spec.builder(n_inputs, n_outputs, **params), layout
