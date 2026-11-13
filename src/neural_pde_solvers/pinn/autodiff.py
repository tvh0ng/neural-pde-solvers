"""Thin wrappers around :func:`torch.autograd.grad` for PDE residuals."""

from __future__ import annotations

import torch


def gradient(u: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Return ``du/dX`` with shape ``(N, d)``; the graph is kept for higher derivatives.

    ``u`` must have shape ``(N, 1)`` and be computed from ``X`` (``X.requires_grad``).
    """
    (du,) = torch.autograd.grad(
        u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True
    )
    return du


def partial(u: torch.Tensor, X: torch.Tensor, *dims: int) -> torch.Tensor:
    """Mixed partial derivative of ``u`` w.r.t. the input columns listed in ``dims``.

    Examples
    --------
    ``partial(u, X, 0)``     -> du/dx
    ``partial(u, X, 0, 0)``  -> d2u/dx2
    ``partial(u, X, 0, 1)``  -> d2u/dxdt
    """
    if not dims:
        return u
    out = u
    for d in dims:
        out = gradient(out, X)[:, d : d + 1]
    return out


def laplacian(
    u: torch.Tensor, X: torch.Tensor, dims: tuple[int, ...] | None = None
) -> torch.Tensor:
    """Sum of second derivatives over ``dims`` (all input columns by default)."""
    dims = tuple(range(X.shape[1])) if dims is None else dims
    du = gradient(u, X)
    return sum(gradient(du[:, d : d + 1], X)[:, d : d + 1] for d in dims)
