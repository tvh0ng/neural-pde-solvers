"""PDE problem definitions.

Each problem is an :class:`torch.nn.Module` so that its fixed training points
(initial/boundary/measurement data) are registered as buffers and move with
``.to(device)``, and so that unknown physical parameters (inverse problems) are
ordinary ``nn.Parameter`` objects picked up by the optimizer.

A problem defines

* the domain ``lb``/``ub`` and input variable names,
* :meth:`sample_collocation` — random interior points where the PDE is enforced,
* :meth:`residual` — the strong-form PDE residual ``R[u](X)``,
* :meth:`losses` — a dict of mean-squared loss terms (``pde``, ``bc``, ``ic``, ``data``),
* :meth:`exact` — a reference solution on NumPy arrays, when available.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from neural_pde_solvers.pinn.autodiff import gradient, laplacian


def _mse(a: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    return torch.mean(a**2) if b is None else torch.mean((a - b) ** 2)


class PDEProblem(nn.Module, ABC):
    """Base class for forward and inverse PDE problems."""

    name: str = "problem"
    input_names: tuple[str, ...] = ()
    default_weights: dict[str, float] = {}

    def __init__(self, lb: Sequence[float], ub: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))

    # ------------------------------------------------------------------ helpers
    @property
    def dim(self) -> int:
        return len(self.input_names)

    @property
    def device(self) -> torch.device:
        return self.lb.device

    def domain(self) -> tuple[list[float], list[float]]:
        return self.lb.tolist(), self.ub.tolist()

    def sample_collocation(self, n: int) -> torch.Tensor:
        """Uniform random points in the box ``[lb, ub]`` with ``requires_grad=True``."""
        X = self.lb + (self.ub - self.lb) * torch.rand(n, self.dim, device=self.device)
        return X.requires_grad_(True)

    # ------------------------------------------------------------ problem API
    @abstractmethod
    def residual(self, model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        """PDE residual at collocation points ``X`` (shape ``(N, 1)``)."""

    @abstractmethod
    def losses(self, model: nn.Module, X_col: torch.Tensor) -> dict[str, torch.Tensor]:
        """Unweighted loss terms."""

    def exact(self, X: np.ndarray) -> np.ndarray | None:  # noqa: ARG002
        """Reference solution at ``X`` (shape ``(N, d)``) or ``None`` if unknown."""
        return None

    def physical_parameters(self) -> dict[str, float]:
        """Current values of identifiable physical parameters (inverse problems)."""
        return {}


# =============================================================================
# 1. Steady 1D heat (Poisson) equation
# =============================================================================
class Heat1D(PDEProblem):
    r"""Steady 1D heat equation  :math:`u''(x) = -\sin(\pi x)` on ``[0, 1]``.

    Boundary conditions ``u(0) = u(1) = 0``; exact solution ``sin(pi x) / pi^2``.
    """

    name = "heat1d"
    input_names = ("x",)
    default_weights = {"pde": 1.0, "bc": 10.0}

    def __init__(self) -> None:
        super().__init__(lb=[0.0], ub=[1.0])
        self.register_buffer("X_bc", torch.tensor([[0.0], [1.0]]))
        self.register_buffer("u_bc", torch.zeros(2, 1))

    @staticmethod
    def source(x: torch.Tensor) -> torch.Tensor:
        return -torch.sin(math.pi * x)

    def residual(self, model, X):
        u = model(X)
        return laplacian(u, X) - self.source(X)

    def losses(self, model, X_col):
        return {
            "pde": _mse(self.residual(model, X_col)),
            "bc": _mse(model(self.X_bc), self.u_bc),
        }

    def exact(self, X):
        return np.sin(np.pi * X[:, :1]) / np.pi**2


# =============================================================================
# 2. Viscous Burgers' equation
# =============================================================================
def burgers_reference(
    x: np.ndarray, t: np.ndarray, nu: float = 0.01 / np.pi, n_quad: int = 4001, chunk: int = 2048
) -> np.ndarray:
    r"""Exact solution of Burgers' equation with ``u(x,0) = -sin(pi x)`` via Cole–Hopf.

    .. math::
        u(x,t) = -\frac{\int \sin\pi(x-\eta)\, f(x-\eta)\, e^{-\eta^2/4\nu t}\,d\eta}
                       {\int f(x-\eta)\, e^{-\eta^2/4\nu t}\,d\eta},
        \qquad f(y) = \exp\!\left(-\frac{\cos \pi y}{2\pi\nu}\right).

    The integrals are evaluated with the trapezoidal rule on ``eta = sqrt(4 nu t) z``,
    ``z in [-10, 10]``, using a log-sum-exp shift to avoid overflow. Boundary values
    ``u(+-1, t) = 0`` are reproduced to quadrature precision.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    t = np.asarray(t, dtype=np.float64).ravel()
    x, t = np.broadcast_arrays(x, t)
    out = np.empty_like(x)
    z = np.linspace(-10.0, 10.0, n_quad)
    for s in range(0, x.size, chunk):
        xs, ts = x[s : s + chunk, None], t[s : s + chunk, None]
        at_zero = ts[:, 0] <= 0.0
        scale = np.sqrt(4.0 * nu * np.where(at_zero[:, None], 1.0, ts))
        y = xs - scale * z[None, :]
        log_w = -np.cos(np.pi * y) / (2.0 * np.pi * nu) - z[None, :] ** 2
        w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
        num = np.trapezoid(np.sin(np.pi * y) * w, z, axis=1)
        den = np.trapezoid(w, z, axis=1)
        out[s : s + chunk] = np.where(at_zero, -np.sin(np.pi * xs[:, 0]), -num / den)
    return out


class Burgers1D(PDEProblem):
    r"""Viscous Burgers' equation :math:`u_t + u u_x = \nu u_{xx}` on ``[-1,1] x [0,1]``.

    ``u(x, 0) = -sin(pi x)``, ``u(-1, t) = u(1, t) = 0``, ``nu = 0.01 / pi`` by default.
    A shock forms at ``x = 0`` around ``t ~ 0.3``.
    """

    name = "burgers1d"
    input_names = ("x", "t")
    default_weights = {"pde": 1.0, "ic": 20.0, "bc": 10.0}

    def __init__(self, nu: float = 0.01 / math.pi, n_ic: int = 200, n_bc: int = 200) -> None:
        super().__init__(lb=[-1.0, 0.0], ub=[1.0, 1.0])
        self.nu = float(nu)
        x_ic = 2.0 * torch.rand(n_ic, 1) - 1.0
        self.register_buffer("X_ic", torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1))
        self.register_buffer("u_ic", -torch.sin(math.pi * x_ic))
        half = n_bc // 2
        x_bc = torch.cat([-torch.ones(half, 1), torch.ones(n_bc - half, 1)])
        self.register_buffer("X_bc", torch.cat([x_bc, torch.rand(n_bc, 1)], dim=1))
        self.register_buffer("u_bc", torch.zeros(n_bc, 1))

    def residual(self, model, X):
        u = model(X)
        du = gradient(u, X)
        u_x, u_t = du[:, 0:1], du[:, 1:2]
        u_xx = gradient(u_x, X)[:, 0:1]
        return u_t + u * u_x - self.nu * u_xx

    def losses(self, model, X_col):
        return {
            "pde": _mse(self.residual(model, X_col)),
            "ic": _mse(model(self.X_ic), self.u_ic),
            "bc": _mse(model(self.X_bc), self.u_bc),
        }

    def exact(self, X):
        return burgers_reference(X[:, 0], X[:, 1], nu=self.nu).reshape(-1, 1)


# =============================================================================
# 3. 2D Poisson equation
# =============================================================================
class Poisson2D(PDEProblem):
    r"""2D Poisson equation :math:`\Delta u = -2\pi^2 \sin(\pi x)\sin(\pi y)` on ``[0,1]^2``.

    Homogeneous Dirichlet conditions on all four edges; exact solution
    ``sin(pi x) sin(pi y)``.
    """

    name = "poisson2d"
    input_names = ("x", "y")
    default_weights = {"pde": 1.0, "bc": 10.0}

    def __init__(self, n_bc: int = 400) -> None:
        super().__init__(lb=[0.0, 0.0], ub=[1.0, 1.0])
        n = n_bc // 4
        s = torch.rand(4, n, 1)
        edges = [
            torch.cat([s[0], torch.zeros(n, 1)], 1),  # bottom  y = 0
            torch.cat([s[1], torch.ones(n, 1)], 1),  # top     y = 1
            torch.cat([torch.zeros(n, 1), s[2]], 1),  # left    x = 0
            torch.cat([torch.ones(n, 1), s[3]], 1),  # right   x = 1
        ]
        self.register_buffer("X_bc", torch.cat(edges, 0))
        self.register_buffer("u_bc", torch.zeros(4 * n, 1))

    @staticmethod
    def source(X: torch.Tensor) -> torch.Tensor:
        return -2.0 * math.pi**2 * torch.sin(math.pi * X[:, 0:1]) * torch.sin(math.pi * X[:, 1:2])

    def residual(self, model, X):
        return laplacian(model(X), X) - self.source(X)

    def losses(self, model, X_col):
        return {
            "pde": _mse(self.residual(model, X_col)),
            "bc": _mse(model(self.X_bc), self.u_bc),
        }

    def exact(self, X):
        return (np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])).reshape(-1, 1)


# =============================================================================
# 4. Inverse problem: identify the diffusion coefficient
# =============================================================================
class DiffusionInverse1D(PDEProblem):
    r"""Identify ``D`` in :math:`u_t = D u_{xx}` on ``[0,1] x [0,1]`` from noisy data.

    ``u(x,0) = sin(pi x)``, ``u(0,t) = u(1,t) = 0``; exact solution
    ``sin(pi x) exp(-D pi^2 t)``. ``D`` is learned as ``exp(log_D)`` so it stays positive.

    Parameters
    ----------
    D_true
        Coefficient used to synthesise measurements (unknown to the optimiser).
    D_init
        Initial guess for ``D``.
    n_data, noise
        Number of measurements and Gaussian noise level relative to ``std(u)``.
    seed
        Seed of the measurement generator (independent of the global torch seed).
    """

    name = "diffusion_inverse1d"
    input_names = ("x", "t")
    default_weights = {"pde": 1.0, "data": 10.0, "ic": 10.0, "bc": 10.0}

    def __init__(
        self,
        D_true: float = 0.1,
        D_init: float = 0.5,
        n_data: int = 200,
        noise: float = 0.02,
        n_ic: int = 200,
        n_bc: int = 200,
        seed: int = 42,
    ) -> None:
        super().__init__(lb=[0.0, 0.0], ub=[1.0, 1.0])
        if D_init <= 0:
            raise ValueError("D_init must be positive.")
        self.D_true, self.D_init, self.noise = float(D_true), float(D_init), float(noise)
        self.log_D = nn.Parameter(torch.tensor(math.log(D_init), dtype=torch.float32))

        rng = np.random.default_rng(seed)
        X_data = rng.uniform(0.0, 1.0, size=(n_data, 2))
        u_data = self._solution(X_data, self.D_true)
        u_data = u_data + noise * u_data.std() * rng.standard_normal(u_data.shape)
        self.register_buffer("X_data", torch.tensor(X_data, dtype=torch.float32))
        self.register_buffer("u_data", torch.tensor(u_data, dtype=torch.float32))

        x_ic = torch.rand(n_ic, 1)
        self.register_buffer("X_ic", torch.cat([x_ic, torch.zeros_like(x_ic)], 1))
        self.register_buffer("u_ic", torch.sin(math.pi * x_ic))
        half = n_bc // 2
        x_bc = torch.cat([torch.zeros(half, 1), torch.ones(n_bc - half, 1)])
        self.register_buffer("X_bc", torch.cat([x_bc, torch.rand(n_bc, 1)], 1))
        self.register_buffer("u_bc", torch.zeros(n_bc, 1))

    @property
    def D(self) -> torch.Tensor:
        return torch.exp(self.log_D)

    @staticmethod
    def _solution(X: np.ndarray, D: float) -> np.ndarray:
        return (np.sin(np.pi * X[:, 0]) * np.exp(-D * np.pi**2 * X[:, 1])).reshape(-1, 1)

    def residual(self, model, X):
        u = model(X)
        du = gradient(u, X)
        u_t = du[:, 1:2]
        u_xx = gradient(du[:, 0:1], X)[:, 0:1]
        return u_t - self.D * u_xx

    def losses(self, model, X_col):
        return {
            "pde": _mse(self.residual(model, X_col)),
            "data": _mse(model(self.X_data), self.u_data),
            "ic": _mse(model(self.X_ic), self.u_ic),
            "bc": _mse(model(self.X_bc), self.u_bc),
        }

    def exact(self, X):
        return self._solution(X, self.D_true)

    def physical_parameters(self) -> dict[str, float]:
        return {"D": float(self.D.detach())}


PROBLEMS: dict[str, type[PDEProblem]] = {
    cls.name: cls for cls in (Heat1D, Burgers1D, Poisson2D, DiffusionInverse1D)
}


def build_problem(name: str, **kwargs) -> PDEProblem:
    if name not in PROBLEMS:
        raise KeyError(f"Unknown problem '{name}'. Available: {sorted(PROBLEMS)}")
    return PROBLEMS[name](**kwargs)
