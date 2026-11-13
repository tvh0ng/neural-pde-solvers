"""Physics-informed neural networks in PyTorch."""

from neural_pde_solvers.pinn.autodiff import gradient, laplacian, partial
from neural_pde_solvers.pinn.experiment import build_model, resolve_device, run_experiment
from neural_pde_solvers.pinn.networks import MLP, count_parameters
from neural_pde_solvers.pinn.problems import (
    PROBLEMS,
    Burgers1D,
    DiffusionInverse1D,
    Heat1D,
    PDEProblem,
    Poisson2D,
    build_problem,
    burgers_reference,
)
from neural_pde_solvers.pinn.trainer import (
    PINNTrainer,
    TrainerConfig,
    error_metrics,
    evaluate,
    grid,
    predict,
)

__all__ = [
    "MLP",
    "PROBLEMS",
    "Burgers1D",
    "DiffusionInverse1D",
    "Heat1D",
    "PDEProblem",
    "PINNTrainer",
    "Poisson2D",
    "TrainerConfig",
    "build_model",
    "build_problem",
    "burgers_reference",
    "count_parameters",
    "error_metrics",
    "evaluate",
    "gradient",
    "grid",
    "laplacian",
    "partial",
    "predict",
    "resolve_device",
    "run_experiment",
]
