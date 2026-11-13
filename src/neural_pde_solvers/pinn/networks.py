"""Neural-network architectures for PINNs.

All networks take a single tensor ``X`` of shape ``(N, d)`` whose columns are the
independent variables (e.g. ``[x, t]``) and return ``u`` of shape ``(N, out_dim)``.
Using one stacked input keeps derivative code generic: ``du/dX[:, k]`` is column ``k``
of the gradient.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class Sine(nn.Module):
    """``sin(x)`` activation (SIREN-style), useful for oscillatory solutions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "sin": Sine,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "softplus": nn.Softplus,
}


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Choose from {sorted(_ACTIVATIONS)}.")
    return _ACTIVATIONS[name]()


class MLP(nn.Module):
    """Fully connected network with optional residual (skip) connections.

    Smooth activations (``tanh`` by default) are used because PDE residuals need
    higher-order derivatives of the output; ReLU's second derivative vanishes almost
    everywhere.

    Parameters
    ----------
    in_dim, out_dim
        Number of input coordinates and output fields.
    width, depth
        Neurons per hidden layer and number of hidden layers.
    activation
        One of ``tanh``, ``sin``, ``silu``, ``gelu``, ``softplus``.
    residual
        If ``True``, hidden layers compute ``h <- act(W h + b) + h``.
    lb, ub
        Optional lower/upper bounds of the domain. When given, inputs are mapped
        affinely to ``[-1, 1]`` before the first layer, which conditions training on
        domains that are far from the unit cube.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        width: int = 64,
        depth: int = 4,
        activation: str = "tanh",
        residual: bool = False,
        lb: Sequence[float] | None = None,
        ub: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.in_dim, self.out_dim, self.residual = in_dim, out_dim, residual

        if (lb is None) != (ub is None):
            raise ValueError("Provide both lb and ub, or neither.")
        if lb is not None:
            lb_t = torch.as_tensor(lb, dtype=torch.float32).reshape(1, in_dim)
            ub_t = torch.as_tensor(ub, dtype=torch.float32).reshape(1, in_dim)
            if torch.any(ub_t <= lb_t):
                raise ValueError("Each ub must be strictly greater than lb.")
            self.register_buffer("lb", lb_t)
            self.register_buffer("ub", ub_t)
        else:
            self.lb = self.ub = None

        self.input_layer = nn.Linear(in_dim, width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 1))
        self.output_layer = nn.Linear(width, out_dim)
        self.act = make_activation(activation)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-normal weights and zero biases (standard choice for tanh PINNs)."""
        for layer in [self.input_layer, *self.hidden, self.output_layer]:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] != self.in_dim:
            raise ValueError(f"Expected input with {self.in_dim} columns, got {X.shape[-1]}.")
        if self.lb is not None:
            X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        h = self.act(self.input_layer(X))
        for layer in self.hidden:
            h_new = self.act(layer(h))
            h = h_new + h if self.residual else h_new
        return self.output_layer(h)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
