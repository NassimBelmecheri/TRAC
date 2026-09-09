# Reproducibility assets

The pretrained **T-ORACLE checkpoints** and the per-variant **datasets** used to
reproduce the paper's tables (the paper's original IJCAI assets, covering all
benchmarks and the TO1/TO2/TO3 variants) are **not** stored in this Git
repository. They are hosted externally and downloaded into a local *assets root*.

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

## Download (Nextcloud)

1. Download the reproducibility assets archive from:

   **https://nextcloud.lisn.upsaclay.fr/index.php/s/DCSAHyt5qpmxK2a**

2. Extract it so the folders above sit directly inside your chosen assets root.

   ```bash
   # option A: extract into this folder (default location)
   unzip trac_reproducibility_assets.zip -d reproducibility

   # option B: extract anywhere and point TRAC_ASSETS at it
   unzip trac_reproducibility_assets.zip -d /path/to/assets
   export TRAC_ASSETS=/path/to/assets           # Windows: setx TRAC_ASSETS "C:\path\to\assets"
   ```

## Reproduce

```bash
python cli.py reproduce --rq 2      # Table 2 (uses the downloaded checkpoints)
python cli.py reproduce --rq 4      # Table 4
```

> If the assets are absent, the platform still works end to end — it will
> **generate** data and **train** fresh models on the fly (see the main README).
> The downloaded assets are only needed to reproduce the paper's *pretrained*
> numbers exactly.
