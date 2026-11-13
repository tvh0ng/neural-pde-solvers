"""Unit tests for the Keras surrogate subpackage."""

import json

import numpy as np
import pytest

pytest.importorskip("tensorflow")

import keras  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

from neural_pde_solvers.surrogate import (  # noqa: E402
    MODELS,
    DataConfig,
    ExperimentConfig,
    R2Score,
    build_model,
    load_trained,
    reshape_inputs,
    run_experiment,
    split_and_scale,
)
from neural_pde_solvers.utils import load_yaml  # noqa: E402

N_IN, N_OUT = 10, 12


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(300, N_IN))
    Y = np.tanh(0.5 * X @ rng.normal(size=(N_IN, N_OUT))) + 5.0
    return np.hstack([X, Y])


def data_cfg(**kw):
    base = dict(path="unused.csv", input_cols=(0, N_IN), output_cols=(N_IN, N_IN + N_OUT))
    return DataConfig(**{**base, **kw})


# -------------------------------------------------------------------- data
def test_split_sizes_and_no_leakage(table):
    s = split_and_scale(table, data_cfg(test_ratio=0.2, val_ratio=0.1))
    assert (len(s.x_train), len(s.x_val), len(s.x_test)) == (216, 24, 60)
    # scaler statistics come from the train block only
    np.testing.assert_allclose(s.scaler_x.data_min_, table[:216, :N_IN].min(axis=0))
    np.testing.assert_allclose(s.x_train.min(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(s.x_train.max(axis=0), 1.0, atol=1e-6)


def test_sequential_split_preserves_row_order(table):
    s = split_and_scale(table, data_cfg(val_ratio=0.0))
    assert s.x_val is None
    back = s.scaler_x.inverse_transform(s.x_test)
    np.testing.assert_allclose(back, table[240:, :N_IN], rtol=1e-5)


def test_config_validation():
    with pytest.raises(ValueError):
        data_cfg(input_cols=(5, 5))
    with pytest.raises(ValueError):
        data_cfg(test_ratio=1.0)
    with pytest.raises(ValueError):
        split_and_scale(np.zeros((10, 5)), data_cfg())


@pytest.mark.parametrize(
    "path", sorted(__import__("pathlib").Path("configs/surrogate").glob("*.yaml"))
)
def test_shipped_configs_parse_and_build(path):
    cfg = ExperimentConfig.from_dict(load_yaml(path))
    model, layout = build_model(
        cfg.model, cfg.data.n_inputs, cfg.data.n_outputs, **cfg.model_params
    )
    x = reshape_inputs(np.zeros((2, cfg.data.n_inputs), np.float32), layout)
    assert model(x).shape == (2, cfg.data.n_outputs)


# ------------------------------------------------------------------ models
@pytest.mark.parametrize("name", sorted(MODELS))
def test_every_model_builds_with_correct_io(name):
    params = (
        {"conv_filters": [8, 8], "conv_kernels": [2, 2]} if name in {"cnn1d", "cnn_gru"} else {}
    )
    if name in {"rnn", "gru", "lstm", "cnn_gru"}:
        params["units"] = [8, 4]
    model, layout = build_model(name, N_IN, N_OUT, **params)
    x = reshape_inputs(np.zeros((3, N_IN), np.float32), layout)
    out = model(x)
    assert out.shape == (3, N_OUT)
    assert model.layers[-1].dtype_policy.compute_dtype == "float32"


def test_layout_override_only_for_recurrent():
    _, layout = build_model("gru", N_IN, N_OUT, units=[4], layout="sequence")
    assert layout == "sequence"
    with pytest.raises(ValueError):
        build_model("ann", N_IN, N_OUT, layout="sequence")


# ----------------------------------------------------------------- metric
def test_streaming_r2_matches_pooled_r2():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(100, 4)).astype("float32")
    p = (y + 0.3 * rng.normal(size=y.shape)).astype("float32")
    m = R2Score()
    for i in range(0, 100, 7):  # uneven batches
        m.update_state(y[i : i + 7], p[i : i + 7])
    pooled = r2_score(y.ravel(), p.ravel())
    assert float(m.result()) == pytest.approx(pooled, abs=1e-4)
    m.reset_state()
    assert float(m.count) == 0.0


# --------------------------------------------------------------- pipeline
def test_end_to_end_training_and_reload(table, tmp_path):
    csv = tmp_path / "data.csv"
    np.savetxt(csv, table, delimiter=",")
    cfg = ExperimentConfig.from_dict(
        {
            "name": "tiny",
            "model": "ann",
            "data": {
                "path": str(csv),
                "input_cols": [0, N_IN],
                "output_cols": [N_IN, N_IN + N_OUT],
            },
            "model_params": {
                "dense_units": [32, 16],
                "dropout_after": [],
                "kernel_initializer": "glorot_uniform",
            },
            "training": {"epochs": 40, "batch_size": 32, "learning_rate": 3e-3, "verbose": 0},
            "output_dir": str(tmp_path / "out"),
        }
    )
    res = run_experiment(cfg)
    files = {p.name for p in (tmp_path / "out").iterdir()}
    assert {"best.keras", "history.csv", "metrics.json", "scalers.joblib", "config.yaml"} <= files
    metrics = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert metrics["test"]["physical"]["r2"] > 0.5
    assert res.history["loss"][-1] < res.history["loss"][0]

    model, scalers, cfg2 = load_trained(tmp_path / "out")
    assert cfg2.model == "ann" and set(scalers) == {"x", "y"}
    raw = table[:5, :N_IN]
    np.testing.assert_allclose(res.predict(raw).shape, (5, N_OUT))
    keras.backend.clear_session()
