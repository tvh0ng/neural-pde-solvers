"""Generic PINN training loop (Adam, optionally followed by L-BFGS)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from neural_pde_solvers.pinn.problems import PDEProblem
from neural_pde_solvers.utils import get_logger

log = get_logger("pinn")


@dataclass
class TrainerConfig:
    """Optimisation settings.

    Attributes
    ----------
    epochs
        Number of Adam iterations (one full-batch step each).
    lr
        Adam learning rate.
    n_collocation
        Number of interior collocation points.
    resample_every
        Draw fresh collocation points every ``k`` epochs (``0`` = never). Resampling
        acts as a cheap stochastic quadrature and reduces over-fitting of the residual.
    scheduler
        ``"none"``, ``"step"`` (``StepLR``) or ``"plateau"`` (``ReduceLROnPlateau``).
    scheduler_kwargs
        Forwarded to the scheduler constructor.
    lbfgs_iters
        Maximum L-BFGS iterations after Adam (``0`` disables). L-BFGS with a strong-Wolfe
        line search typically lowers the error by an order of magnitude once Adam has
        found the right basin.
    log_every
        Logging interval in epochs (``0`` silences progress logs).
    grad_clip
        Optional max gradient norm for Adam steps.
    """

    epochs: int = 5000
    lr: float = 1e-3
    n_collocation: int = 1000
    resample_every: int = 1
    scheduler: str = "none"
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    lbfgs_iters: int = 0
    log_every: int = 500
    grad_clip: float | None = None


class PINNTrainer:
    """Train ``model`` so that it satisfies ``problem``.

    Parameters
    ----------
    model
        Network mapping ``(N, d)`` coordinates to ``(N, 1)`` fields.
    problem
        A :class:`PDEProblem`; its own ``nn.Parameter`` objects (e.g. an unknown
        diffusion coefficient) are optimised jointly with the network weights.
    config
        :class:`TrainerConfig`.
    weights
        Loss weights; missing keys fall back to ``problem.default_weights``.
    """

    def __init__(
        self,
        model: nn.Module,
        problem: PDEProblem,
        config: TrainerConfig | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.model, self.problem = model, problem
        self.config = config or TrainerConfig()
        self.weights = {**problem.default_weights, **(weights or {})}
        self.params = list(model.parameters()) + list(problem.parameters())
        self.history: dict[str, list[float]] = {}

    # ----------------------------------------------------------------- internals
    def _loss(self, X_col: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        terms = self.problem.losses(self.model, X_col)
        unknown = set(terms) - set(self.weights)
        if unknown:
            raise KeyError(f"No weight given for loss terms {sorted(unknown)}")
        total = sum(self.weights[k] * v for k, v in terms.items())
        return total, terms

    def _record(self, epoch: int, total: torch.Tensor, terms: dict, lr: float) -> None:
        row = {"epoch": epoch, "lr": lr, "total": total.item()}
        row.update({k: v.item() for k, v in terms.items()})
        row.update(self.problem.physical_parameters())
        for k, v in row.items():
            self.history.setdefault(k, []).append(float(v))

    def _log(self, stage: str, epoch: int, total_epochs: int) -> None:
        h = self.history
        parts = [f"{stage} {epoch:>6d}/{total_epochs}", f"loss {h['total'][-1]:.3e}"]
        parts += [f"{k} {h[k][-1]:.2e}" for k in self.weights if k in h]
        parts += [f"{k} {h[k][-1]:.5f}" for k in self.problem.physical_parameters()]
        parts.append(f"lr {h['lr'][-1]:.1e}")
        log.info(" | ".join(parts))

    def _make_scheduler(self, optimizer: torch.optim.Optimizer):
        kind, kw = self.config.scheduler.lower(), dict(self.config.scheduler_kwargs)
        if kind == "none":
            return None
        if kind == "step":
            return torch.optim.lr_scheduler.StepLR(optimizer, **kw)
        if kind == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kw)
        raise ValueError(f"Unknown scheduler '{self.config.scheduler}'")

    # ----------------------------------------------------------------- public API
    def fit(self) -> dict[str, list[float]]:
        """Run Adam (then L-BFGS if configured) and return the loss history."""
        cfg = self.config
        self.model.train()
        optimizer = torch.optim.Adam(self.params, lr=cfg.lr)
        scheduler = self._make_scheduler(optimizer)
        X_col = self.problem.sample_collocation(cfg.n_collocation)
        t0 = time.perf_counter()

        for epoch in range(1, cfg.epochs + 1):
            if cfg.resample_every and epoch > 1 and (epoch - 1) % cfg.resample_every == 0:
                X_col = self.problem.sample_collocation(cfg.n_collocation)
            optimizer.zero_grad(set_to_none=True)
            total, terms = self._loss(X_col)
            total.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(self.params, cfg.grad_clip)
            optimizer.step()
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(total.item())
            elif scheduler is not None:
                scheduler.step()
            self._record(epoch, total, terms, optimizer.param_groups[0]["lr"])
            if cfg.log_every and (epoch == 1 or epoch % cfg.log_every == 0):
                self._log("adam", epoch, cfg.epochs)

        if cfg.lbfgs_iters > 0:
            self._fit_lbfgs(X_col, start_epoch=cfg.epochs)

        self.train_time = time.perf_counter() - t0
        log.info("Training finished in %.1f s", self.train_time)
        self.model.eval()
        return self.history

    def _fit_lbfgs(self, X_col: torch.Tensor, start_epoch: int) -> None:
        cfg = self.config
        optimizer = torch.optim.LBFGS(
            self.params,
            lr=1.0,
            max_iter=cfg.lbfgs_iters,
            max_eval=int(cfg.lbfgs_iters * 1.25),
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        it = [start_epoch]

        def closure():
            optimizer.zero_grad(set_to_none=True)
            total, terms = self._loss(X_col)
            total.backward()
            it[0] += 1
            self._record(it[0], total, terms, 1.0)
            if cfg.log_every and (it[0] - start_epoch) % cfg.log_every == 0:
                self._log("lbfgs", it[0] - start_epoch, cfg.lbfgs_iters)
            return total

        optimizer.step(closure)


# =============================================================================
# Evaluation helpers
# =============================================================================
@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int = 65536) -> np.ndarray:
    """Evaluate ``model`` on a NumPy array of coordinates, returning a NumPy array."""
    model.eval()
    device = next(model.parameters()).device
    out = [
        model(torch.as_tensor(X[i : i + batch_size], dtype=torch.float32, device=device)).cpu()
        for i in range(0, len(X), batch_size)
    ]
    return torch.cat(out).numpy()


def grid(
    problem: PDEProblem, n: int | tuple[int, ...] = 101
) -> tuple[list[np.ndarray], np.ndarray]:
    """Tensor-product grid over the problem domain.

    Returns the list of meshgrid arrays (``indexing="xy"``) and the stacked ``(N, d)`` points.
    """
    lb, ub = problem.domain()
    ns = (n,) * problem.dim if isinstance(n, int) else n
    axes = [np.linspace(a, b, k) for a, b, k in zip(lb, ub, ns, strict=True)]
    mesh = np.meshgrid(*axes)
    return mesh, np.stack([m.ravel() for m in mesh], axis=1)


def error_metrics(pred: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    """Relative L2, max-absolute and mean-absolute errors."""
    pred, exact = np.asarray(pred).ravel(), np.asarray(exact).ravel()
    err = pred - exact
    return {
        "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(exact)),
        "max_abs": float(np.abs(err).max()),
        "mean_abs": float(np.abs(err).mean()),
    }


def evaluate(
    model: nn.Module, problem: PDEProblem, n: int | tuple[int, ...] = 101
) -> dict[str, float]:
    """Compare the network with ``problem.exact`` on a uniform grid."""
    _, X = grid(problem, n)
    exact = problem.exact(X)
    if exact is None:
        raise ValueError(f"Problem '{problem.name}' has no reference solution.")
    metrics = error_metrics(predict(model, X), exact)
    metrics.update(problem.physical_parameters())
    return metrics
