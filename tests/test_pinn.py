"""Unit tests for the PyTorch PINN subpackage."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neural_pde_solvers.pinn import (  # noqa: E402
    MLP,
    Burgers1D,
    DiffusionInverse1D,
    Heat1D,
    PINNTrainer,
    Poisson2D,
    TrainerConfig,
    burgers_reference,
    error_metrics,
    evaluate,
    grid,
    laplacian,
    partial,
)
from neural_pde_solvers.utils import set_seed  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    set_seed(0)


class Analytic(torch.nn.Module):
    """Wrap a closed-form u(X) so it can be plugged into a problem's residual."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, X):
        return self.fn(X)


# --------------------------------------------------------------------- network
def test_mlp_shapes_and_normalisation():
    net = MLP(2, 1, width=16, depth=3, lb=[-1, 0], ub=[1, 2])
    assert net(torch.rand(7, 2)).shape == (7, 1)
    with pytest.raises(ValueError):
        net(torch.rand(7, 3))
    with pytest.raises(ValueError):
        MLP(2, lb=[0, 0], ub=[1, 0])


def test_residual_mlp_runs():
    net = MLP(2, 1, width=8, depth=4, residual=True)
    assert net(torch.rand(5, 2)).shape == (5, 1)


# -------------------------------------------------------------------- autodiff
def test_derivatives_match_closed_form():
    X = torch.rand(50, 2, dtype=torch.float64, requires_grad=True)
    u = X[:, :1] ** 3 * X[:, 1:] ** 2
    x, t = X[:, :1].detach(), X[:, 1:].detach()
    torch.testing.assert_close(partial(u, X, 0), 3 * x**2 * t**2)
    torch.testing.assert_close(partial(u, X, 0, 0), 6 * x * t**2)
    torch.testing.assert_close(partial(u, X, 0, 1), 6 * x**2 * t)
    torch.testing.assert_close(laplacian(u, X), 6 * x * t**2 + 2 * x**3)


# ------------------------------------------------- residuals vanish on exact u
def _max_residual(problem, model, n=256):
    X = problem.sample_collocation(n)
    return problem.residual(model, X).abs().max().item()


def test_heat_exact_solution_has_zero_residual():
    p = Heat1D()
    exact = Analytic(lambda X: torch.sin(math.pi * X) / math.pi**2)
    assert _max_residual(p, exact) < 1e-5


def test_poisson_exact_solution_has_zero_residual():
    p = Poisson2D()
    exact = Analytic(lambda X: torch.sin(math.pi * X[:, :1]) * torch.sin(math.pi * X[:, 1:]))
    assert _max_residual(p, exact) < 1e-4
    assert np.allclose(p.exact(p.X_bc.numpy()), 0.0, atol=1e-6)


def test_inverse_exact_solution_has_zero_residual_at_true_D():
    p = DiffusionInverse1D(D_true=0.1, D_init=0.1)
    exact = Analytic(
        lambda X: torch.sin(math.pi * X[:, :1]) * torch.exp(-0.1 * math.pi**2 * X[:, 1:])
    )
    assert _max_residual(p, exact) < 1e-5
    assert p.physical_parameters()["D"] == pytest.approx(0.1, rel=1e-6)
    assert [n for n, _ in p.named_parameters()] == ["log_D"]


def test_burgers_reference_initial_boundary_and_pde():
    nu = 0.01 / np.pi
    x = np.linspace(-1, 1, 21)
    np.testing.assert_allclose(
        burgers_reference(x, np.zeros_like(x)), -np.sin(np.pi * x), atol=1e-12
    )
    t = np.linspace(0.05, 1, 10)
    np.testing.assert_allclose(burgers_reference(np.ones_like(t), t), 0.0, atol=1e-10)
    # finite-difference check of u_t + u u_x - nu u_xx away from the shock
    x, t, h = np.linspace(-0.9, -0.2, 8), np.full(8, 0.3), 1e-4
    u = burgers_reference(x, t)
    u_t = (burgers_reference(x, t + h) - burgers_reference(x, t - h)) / (2 * h)
    u_x = (burgers_reference(x + h, t) - burgers_reference(x - h, t)) / (2 * h)
    u_xx = (burgers_reference(x + h, t) - 2 * u + burgers_reference(x - h, t)) / h**2
    assert np.abs(u_t + u * u_x - nu * u_xx).max() < 1e-5


def test_problem_buffers_follow_device_and_dtype():
    p = Burgers1D(n_ic=10, n_bc=10)
    assert p.X_ic.shape == (10, 2) and p.X_bc.shape == (10, 2)
    assert torch.all(p.X_ic[:, 1] == 0)
    assert set(p.X_bc[:, 0].tolist()) == {-1.0, 1.0}
    X = p.sample_collocation(100)
    assert X.requires_grad and torch.all(X[:, 0] >= -1) and torch.all(X[:, 1] <= 1)


# --------------------------------------------------------------------- trainer
def test_trainer_reduces_loss_and_solves_heat_equation():
    p = Heat1D()
    model = MLP(1, 1, width=32, depth=3)
    cfg = TrainerConfig(epochs=1500, lr=3e-3, n_collocation=64, lbfgs_iters=200, log_every=0)
    hist = PINNTrainer(model, p, cfg).fit()
    assert hist["total"][-1] < 1e-2 * hist["total"][0]
    assert evaluate(model, p, 201)["rel_l2"] < 1e-2


def test_inverse_parameter_is_optimised():
    p = DiffusionInverse1D(D_init=0.5)
    model = MLP(2, 1, width=16, depth=2)
    hist = PINNTrainer(model, p, TrainerConfig(epochs=50, n_collocation=128, log_every=0)).fit()
    assert "D" in hist and hist["D"][-1] != pytest.approx(0.5)


def test_unknown_weight_key_raises():
    p = Heat1D()
    trainer = PINNTrainer(MLP(1, 1, width=4, depth=1), p, TrainerConfig(epochs=1, log_every=0))
    trainer.weights.pop("bc")
    with pytest.raises(KeyError):
        trainer.fit()


def test_grid_and_error_metrics():
    mesh, X = grid(Poisson2D(), (5, 3))
    assert X.shape == (15, 2) and mesh[0].shape == (3, 5)
    m = error_metrics(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
    assert m["max_abs"] == 1.0 and m["rel_l2"] == pytest.approx(1 / math.sqrt(2))
