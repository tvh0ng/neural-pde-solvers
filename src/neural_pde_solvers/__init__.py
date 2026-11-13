"""neural-pde-solvers — physics-informed neural networks and deep-learning surrogate models.

Subpackages
-----------
neural_pde_solvers.pinn
    PyTorch implementation of forward and inverse PINNs (heat, Burgers, Poisson, diffusion).
neural_pde_solvers.surrogate
    Keras implementation of data-driven surrogate regressors (ANN, 1D-CNN, RNN, GRU, LSTM, CNN-GRU).

The two subpackages are independent: installing ``neural-pde-solvers[pinn]`` does not pull in
TensorFlow, and ``neural-pde-solvers[surrogate]`` does not pull in PyTorch.
"""

from neural_pde_solvers.utils import set_seed, setup_logging

__all__ = ["set_seed", "setup_logging"]
__version__ = "0.2.0"
