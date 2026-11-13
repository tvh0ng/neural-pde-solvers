"""Typed experiment configuration for surrogate models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from neural_pde_solvers.utils import load_yaml


def _from_dict(cls, data: dict[str, Any] | None):
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


@dataclass
class DataConfig:
    """Where the table lives and how it is split.

    The CSV has no header. ``input_cols`` and ``output_cols`` are half-open column
    ranges ``[start, stop)`` (Python slicing), e.g. ``[0, 5]`` = columns 0..4.

    Splitting (``shuffle: false`` keeps the original row order)::

        |<------------- train -------------->|<-- val -->|<---- test ---->|
        |        (1 - test_ratio) * N  (val = last val_ratio of it)  test_ratio * N

    Scalers are fitted on the *train* block only, so neither validation nor test
    statistics leak into training. Early stopping and checkpoint selection use the
    validation block; the test block is touched only once, at the end.
    """

    path: str
    input_cols: tuple[int, int]
    output_cols: tuple[int, int]
    n_samples: int | None = None
    test_ratio: float = 0.2
    val_ratio: float = 0.1
    shuffle: bool = False
    scaler: str = "minmax"

    def __post_init__(self) -> None:
        self.input_cols = tuple(self.input_cols)
        self.output_cols = tuple(self.output_cols)
        for name in ("input_cols", "output_cols"):
            a, b = getattr(self, name)
            if not 0 <= a < b:
                raise ValueError(f"{name} must be [start, stop) with 0 <= start < stop")
        if not 0 < self.test_ratio < 1:
            raise ValueError("test_ratio must be in (0, 1)")
        if not 0 <= self.val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1)")
        if self.scaler not in {"minmax", "standard"}:
            raise ValueError("scaler must be 'minmax' or 'standard'")

    @property
    def n_inputs(self) -> int:
        return self.input_cols[1] - self.input_cols[0]

    @property
    def n_outputs(self) -> int:
        return self.output_cols[1] - self.output_cols[0]


@dataclass
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-3
    patience_es: int = 50
    patience_lr: int = 20
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    mixed_precision: bool | str = "auto"  # True | False | "auto" (= only on GPU)
    verbose: int = 2


@dataclass
class ExperimentConfig:
    name: str
    model: str
    data: DataConfig
    model_params: dict[str, Any] = field(default_factory=dict)
    training: TrainConfig = field(default_factory=TrainConfig)
    output_dir: str = "results/surrogate"
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        d = dict(d)
        d["data"] = _from_dict(DataConfig, d.get("data"))
        d["training"] = _from_dict(TrainConfig, d.get("training"))
        return _from_dict(cls, d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["data"]["input_cols"] = list(self.data.input_cols)
        d["data"]["output_cols"] = list(self.data.output_cols)
        return d
