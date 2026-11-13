"""Command-line entry point: ``surrogate-train configs/surrogate/lstm.yaml --data-path data.csv``."""

from __future__ import annotations

import argparse

from neural_pde_solvers.utils import deep_update, load_yaml, setup_logging


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a deep-learning surrogate model.")
    p.add_argument("config", help="YAML experiment config (see configs/surrogate/).")
    p.add_argument("--data-path", help="Override data.path.")
    p.add_argument("-o", "--output-dir", help="Override output_dir.")
    p.add_argument("--epochs", type=int, help="Override training.epochs.")
    p.add_argument("--seed", type=int)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging()
    from neural_pde_solvers.surrogate.config import ExperimentConfig
    from neural_pde_solvers.surrogate.train import run_experiment

    raw = load_yaml(args.config)
    overrides: dict = {"data": {}, "training": {}}
    if args.data_path:
        overrides["data"]["path"] = args.data_path
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    if args.seed is not None:
        overrides["seed"] = args.seed
    run_experiment(ExperimentConfig.from_dict(deep_update(raw, overrides)))


if __name__ == "__main__":
    main()
