"""Generate a synthetic header-less CSV with the same layout as the lab datasets.

Useful for trying the surrogate pipeline (and for CI) without the private data::

    python scripts/make_synthetic_data.py data/synthetic.csv --rows 2000
    surrogate-train configs/surrogate/lstm.yaml --data-path data/synthetic.csv --epochs 5

Each row holds ``n_in`` random design parameters followed by ``n_out`` values of a
smooth response curve ``y_k = f(params, s_k)`` sampled on ``s in [0, 1]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_table(rows: int, n_in: int = 10, n_out: int = 206, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, size=(rows, n_in))
    s = np.linspace(0.0, 1.0, n_out)[None, :]
    a, b, c, d = (p[:, i : i + 1] for i in range(4))
    rest = p[:, 4:].mean(axis=1, keepdims=True)
    y = (1 + a) * np.sin(np.pi * (1 + 2 * b) * s) * np.exp(-2 * c * s) + d * s**2 + 0.5 * rest
    return np.hstack([p, y])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("output", type=Path)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--n-in", type=int, default=10)
    ap.add_argument("--n-out", type=int, default=206)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = make_table(args.rows, args.n_in, args.n_out, args.seed)
    np.savetxt(args.output, table, delimiter=",", fmt="%.8g")
    print(f"Wrote {table.shape[0]} x {table.shape[1]} table to {args.output}")


if __name__ == "__main__":
    main()
