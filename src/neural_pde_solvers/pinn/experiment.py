"""Config-driven PINN experiments (used by the CLI and the notebooks)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neural_pde_solvers.pinn.networks import MLP
from neural_pde_solvers.pinn.problems import PDEProblem, build_problem
from neural_pde_solvers.pinn.trainer import PINNTrainer, TrainerConfig, evaluate
from neural_pde_solvers.utils import get_logger, save_yaml, set_seed

log = get_logger("pinn")


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(problem: PDEProblem, network: dict[str, Any]) -> MLP:
    """Build an :class:`MLP` for ``problem`` from a ``network`` config section."""
    net = dict(network)
    normalize = net.pop("normalize_inputs", False)
    lb, ub = problem.domain() if normalize else (None, None)
    return MLP(in_dim=problem.dim, out_dim=1, lb=lb, ub=ub, **net)


@dataclass
class PINNResult:
    model: MLP
    problem: PDEProblem
    history: dict[str, list[float]]
    metrics: dict[str, float]


def run_experiment(config: dict[str, Any], output_dir: str | Path | None = None) -> PINNResult:
    """Build, train and evaluate a PINN described by ``config``.

    Expected keys: ``problem`` (``name`` + ``params``), ``network``, ``trainer``,
    optional ``weights``, ``seed``, ``device`` and ``eval_grid``. See ``configs/pinn``.
    """
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    log.info("Device: %s", device)

    pcfg = config["problem"]
    problem = build_problem(pcfg["name"], **pcfg.get("params", {})).to(device)
    model = build_model(problem, config.get("network", {})).to(device)
    trainer = PINNTrainer(
        model, problem, TrainerConfig(**config.get("trainer", {})), config.get("weights")
    )
    history = trainer.fit()
    metrics = evaluate(model, problem, config.get("eval_grid", 101))
    metrics["train_time_s"] = trainer.train_time
    log.info("Metrics: %s", json.dumps(metrics, indent=None))

    if output_dir is not None:
        save_artifacts(Path(output_dir), config, model, history, metrics)
    return PINNResult(model, problem, history, metrics)


def save_artifacts(out: Path, config, model, history, metrics) -> None:
    out.mkdir(parents=True, exist_ok=True)
    save_yaml(config, out / "config.yaml")
    torch.save(model.state_dict(), out / "model.pt")
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    keys = list(history)
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(zip(*(history[k] for k in keys), strict=True))
    log.info("Artifacts written to %s", out)
