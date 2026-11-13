"""Data-driven surrogate regressors in Keras (TensorFlow backend)."""

from neural_pde_solvers.surrogate.config import DataConfig, ExperimentConfig, TrainConfig
from neural_pde_solvers.surrogate.data import (
    DatasetSplits,
    load_splits,
    load_table,
    split_and_scale,
)
from neural_pde_solvers.surrogate.metrics import R2Score, regression_report
from neural_pde_solvers.surrogate.models import MODELS, build_model, reshape_inputs
from neural_pde_solvers.surrogate.train import SurrogateResult, load_trained, run_experiment

__all__ = [
    "MODELS",
    "DataConfig",
    "DatasetSplits",
    "ExperimentConfig",
    "R2Score",
    "SurrogateResult",
    "TrainConfig",
    "build_model",
    "load_splits",
    "load_table",
    "load_trained",
    "regression_report",
    "reshape_inputs",
    "run_experiment",
    "split_and_scale",
]
