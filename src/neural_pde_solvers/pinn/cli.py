"""Command-line entry point: ``pinn-run configs/pinn/burgers1d.yaml``."""

from __future__ import annotations

import argparse

from neural_pde_solvers.utils import deep_update, load_yaml, setup_logging


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a physics-informed neural network.")
    p.add_argument("config", help="YAML experiment config (see configs/pinn/).")
    p.add_argument("-o", "--output-dir", default=None, help="Where to save artifacts.")
    p.add_argument("--epochs", type=int, help="Override trainer.epochs.")
    p.add_argument("--lbfgs-iters", type=int, help="Override trainer.lbfgs_iters.")
    p.add_argument("--device", help="cpu | cuda | auto")
    p.add_argument("--seed", type=int)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    from neural_pde_solvers.pinn.experiment import run_experiment  # heavy import after arg parsing

    config = load_yaml(args.config)
    overrides: dict = {"trainer": {}}
    if args.epochs is not None:
        overrides["trainer"]["epochs"] = args.epochs
    if args.lbfgs_iters is not None:
        overrides["trainer"]["lbfgs_iters"] = args.lbfgs_iters
    if args.device:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = deep_update(config, overrides)
    out = args.output_dir or f"results/pinn/{config['problem']['name']}"
    run_experiment(config, output_dir=out)


if __name__ == "__main__":
    main()
