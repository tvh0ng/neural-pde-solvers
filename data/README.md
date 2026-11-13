# Data

The surrogate datasets are **not** stored in this repository. Place them here (or pass
`--data-path` / set `data.path` in the config).

## Expected format

* Plain CSV, **no header row**, numeric values only.
* One row per sample. The first columns are the input (design) parameters, followed by
  the discretised response. Column ranges are set per experiment in
  `configs/surrogate/*.yaml` as half-open `[start, stop)` intervals:

| Config | File | Rows used | Inputs | Outputs |
|---|---|---|---|---|
| `ann.yaml` | `Out_ML_cm6.csv` | 10 000 | `[0, 5)` | `[6, 206)` |
| `cnn1d.yaml` | `Out_ML_cm.csv` | 30 000 | `[0, 9)` | `[10, 212)` |
| `cnn_gru.yaml`, `lstm.yaml` | `Out_ML_cm.csv` | 30 000 | `[0, 10)` | `[10, 216)` |
| `gru.yaml`, `rnn.yaml` | `Out_ML_cm.csv` | 30 000 | `[0, 5)` | `[6, 206)` |

## Synthetic data

To try the pipeline without the lab data:

```bash
python scripts/make_synthetic_data.py data/synthetic.csv --rows 3000
surrogate-train configs/surrogate/lstm.yaml --data-path data/synthetic.csv
```
