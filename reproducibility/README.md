# Reproducibility assets

The pretrained **T-ORACLE checkpoints** and the per-variant **datasets** used to
reproduce the paper's tables (the paper's original IJCAI assets, covering all
benchmarks and the TO1/TO2/TO3 variants) are **not** stored in this Git
repository. They are provided separately and placed into a local *assets root*.

## Where to put them

The platform looks for the assets under a single root directory, resolved in this
order:

1. the `TRAC_ASSETS` environment variable, if set;
2. otherwise `./reproducibility` (this folder).

Expected layout under that root:

```
<assets root>/
├── conacq_datasets/         # TO1 datasets  (sudoku_9.csv, murder.csv, …)
├── TO1/models/              # TO1 checkpoints (best_model_<name>.csv_<n>_0.0.pth)
├── TO2/                     # TO2 datasets  (dataset_<name>.csv)
├── TO2/models/              # TO2 checkpoints
├── TO3/                     # TO3 datasets  (dataset_<name>_0.8.csv)
└── TO3/models/              # TO3 checkpoints
```

## Providing the assets

These assets are **optional** — with `--source train` the platform generates the
datasets and trains the oracles from scratch, so no download is required (see the
main README).

To reproduce the *pretrained* numbers exactly, obtain the paper's original IJCAI
checkpoints and datasets, arrange them under your assets root using the layout
above, and point `TRAC_ASSETS` at it:

```bash
export TRAC_ASSETS=/path/to/assets           # Windows: $env:TRAC_ASSETS="C:\path\to\assets"
```

## Reproduce

```bash
python cli.py reproduce --rq 2 --source pretrained   # Table 2 (uses the checkpoints)
python cli.py reproduce --rq 4                        # Table 4
python cli.py reproduce --rq 2 --source train         # or regenerate + retrain, no assets
```

> If the assets are absent, the platform still works end to end — it will
> **generate** data and **train** fresh models on the fly (see the main README).
> The assets are only needed to reproduce the paper's *pretrained* numbers exactly.
