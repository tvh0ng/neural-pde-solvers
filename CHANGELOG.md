# Changelog

## 0.2.0 — Restructured as an installable package

- Project renamed from `PINN_DL_ML` to **neural-pde-solvers** (`import neural_pde_solvers`).
  The old GitHub URL redirects to the new one.

### Structure
- All model, training and evaluation code moved out of notebooks into the
  `neural_pde_solvers` package (`src/` layout, `pyproject.toml`, optional extras
  `[pinn]`, `[surrogate]`, `[dev]`).
- Hyperparameters moved to YAML files in `configs/`; two CLIs added
  (`pinn-run`, `surrogate-train`).
- Six near-identical surrogate notebooks (~90 % duplicated code) replaced by one
  `notebooks/surrogate/train_surrogate.ipynb` plus one config per model. Every
  original architecture and hyperparameter is reproduced exactly in
  `configs/surrogate/*.yaml`.
- PINN tutorials renamed `01_…` – `04_…` and rewritten on top of the package; the
  explanations are kept.
- Unit tests (`tests/`), GitHub Actions CI (lint, tests, notebook execution),
  `ruff`, pre-commit hooks and a `Makefile`.

### Fixes — surrogate models
- **Test-set leakage in model selection.** Early stopping, LR scheduling and
  checkpointing previously monitored the *test* set, so reported test scores were
  optimistic. A validation block (`val_ratio`, default 10 % of the training rows) is
  now used; the test block is touched only for the final report.
- Metrics are reported in **physical units** (inverse-transformed) as well as in the
  scaled space; scalers are saved with the model (`scalers.joblib`).
- Mixed precision is enabled only when a GPU is present (`mixed_precision: auto`);
  forcing `mixed_float16` on CPU was slow.
- Keras 3 compatibility: `keras.Input` instead of the deprecated `input_shape=` layer
  argument, learning-rate history key `learning_rate` (was `lr`, so the LR plot was
  always empty), custom `R2Score` registered for serialisation so saved models reload.
- Removed the meaningless `R² × 100 %` line from the metrics file.
- Seeds are set for Python, NumPy and TensorFlow; dataset shuffling is seeded.
- The LSTM notebook used hard-coded file names instead of `model_name`; all
  artifacts now follow one naming scheme.

### Fixes — PINNs
- `ReduceLROnPlateau(verbose=True)` removed (argument no longer exists in recent PyTorch).
- The Burgers notebook claimed input normalisation but did not apply it; `MLP` now
  supports an affine map to `[-1, 1]^d` (`normalize_inputs: true`).
- Burgers' equation now has a **Cole–Hopf reference solution**, so the error is
  measured instead of only plotted.
- Optional **L-BFGS** refinement after Adam (enabled for Burgers).
- Residual definitions are unit-tested against the analytic solutions.
