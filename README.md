# TRAC

**TRAC** (**T**ransformer-o**RA**cle + FAST**C**A) is an end-to-end neuro-symbolic
**constraint-acquisition** platform. A Transformer neural oracle (**T-ORACLE**)
learns to classify (partial) assignments as valid/invalid, and a time-efficient
symbolic learner (**FASTCA**) queries that oracle to reconstruct a constraint
network — all from a single tool with a web UI, a CLI and a Python API.

It implements the paper *"Learning Symbolic Constraint Representations from
Examples: A Neuro-Symbolic Approach."*

```
   ① Generate data ──▶ ② Train T-ORACLE ──▶ ③ Predict / ④ Acquire (FASTCA)
```

Everything is driven from each benchmark's ground-truth oracle, so the whole
pipeline runs **without any external data files**.

---

## Installation

```bash
git clone https://github.com/NassimBelmecheri/TRAC.git
cd TRAC
python -m pip install -r requirements.txt
```

Requires Python 3.10+. The symbolic CA engine (`pycona`) is bundled in the repo
and imported automatically — no separate install needed.

## Quick start — Web UI

```bash
streamlit run app.py
```

Use the tabs left to right: **① Generate → ② Train → ③ Predict → ④ Acquire →
⑤ Reproduce**.

## Command line

```bash
python cli.py list                                   # benchmarks / variants / learners

# full pipeline in one command
python cli.py run --benchmark sudoku --scale demo --variant TO3 --epochs 60

# or step by step
python cli.py generate --benchmark murder --variant TO3 --samples 3000
python cli.py train    --dataset artifacts/data/murder_paper_TO3.csv --epochs 100
python cli.py predict  --model artifacts/models/toracle_murder_TO3.pth
python cli.py acquire  --model artifacts/models/toracle_murder_TO3.pth --learner FASTCA
```

## Python API

```python
from trac import generate_dataset, train_oracle, classify_dataset, acquire

df, meta = generate_dataset("murder", variant="TO3", n_samples=3000)
res      = train_oracle(df, benchmark="murder", scale="paper", variant="TO3", epochs=100)
metrics  = classify_dataset(res["model_path"], df)          # classification metrics
out      = acquire("murder", scale="paper",
                   model_path=res["model_path"], learner="FASTCA")
print(out["evaluation"])   # {'precision': ..., 'recall': ..., 'f1': ...}
```

---

## The pipeline

| Stage | Module | What it does |
|-------|--------|--------------|
| ① Generate | `trac/datagen.py` | Builds labelled datasets for the **TO1/TO2/TO3** variants directly from a benchmark's ground-truth oracle. |
| ② Train | `trac/train.py` | Trains the **T-ORACLE** (`trac/model.py`), sizing the network from the data and saving a self-describing checkpoint. |
| ③ Predict | `trac/predict.py` | Classifies assignments as valid/invalid, with overall and per-arity metrics. |
| ④ Acquire | `trac/acquire.py` | Runs a symbolic learner (**FASTCA** by default) driven by the neural oracle and evaluates the acquired network against ground truth. |

### The T-ORACLE (neural oracle)

A Transformer (`trac/model.py`, `CSPAttentionModel`) that classifies complete or
partial assignments. It combines value + positional embeddings, a transformer
encoder, a learned variable-dependency matrix and a gated output. Three
**variants** differ only in their training data:

- **TO1** — full assignments only.
- **TO2** — a mix of full and partial assignments of varying scope sizes.
- **TO3** — constraint-specific tuples restricted to each target scope
  (the strongest variant).

### FASTCA (symbolic learner)

`trac/fastca.py` exposes `FastCA`, the time-efficient acquisition algorithm. It
optimizes total acquisition time and per-query latency rather than the raw number
of queries, which suits fast machine/neural oracles. Other learners
(`QuAcq`, `MQuAcq2`, `GrowAcq`) are available from the bundled `pycona` engine.

### Neural oracle wrapper

`trac/oracle.py` (`TransformerOracle`) adapts a trained T-ORACLE to the CA oracle
interface: it encodes each membership query into the model's fixed-length input
(each value at its own variable position) and returns the model's yes/no answer,
letting FASTCA run with the neural oracle in place of a human.

---

## Benchmarks, variants, learners

- **Benchmarks** (`trac/benchmarks.py`): `sudoku`, `jsudoku`, `latin_squares`,
  `nqueens`, `job_shop`, `murder`, `exam_timetabling`, `nurse_rostering`,
  `zebra`, `golomb`, `random122`, `random495`. Each exposes a fast **`demo`**
  scale and a **`paper`** scale (paper-sized instances). Model dimensions are
  derived from the constructed instance.
- **Variants**: `TO1`, `TO2`, `TO3`.
- **Learners**: `FASTCA` (default), `QuAcq`, `MQuAcq2`, `GrowAcq`.
- **Evaluation**: constraint-level precision/recall — a learned constraint is
  correct iff logically implied by the target network `T`; a target constraint is
  recovered iff implied by the learned network `L`.

---

## Reproducing the paper (RQ1–RQ4)

The **⑤ Reproduce** UI tab and the `reproduce` CLI command regenerate the paper's
four result tables over the paper's 10 benchmarks.

```bash
python cli.py reproduce --rq 2                       # Table 2 (oracle classification)
python cli.py reproduce --rq 1 --benchmarks murder,zebra   # Table 1 (learner efficiency)
python cli.py reproduce --rq 3 --benchmarks murder,zebra   # Table 3 (learner × oracle)
python cli.py reproduce --rq 4 --constraints "X!=Y,X+Y>Z"  # Table 4 (constraint checker)
```

| RQ | Question | Table | Function |
|----|----------|-------|----------|
| RQ1 | Does FASTCA cut total acquisition time vs query-minimizing learners? | 1 | `trac.rq1` |
| RQ2 | How well do TO1/TO2/TO3 classify (partial) assignments? | 2 | `trac.rq2` |
| RQ3 | How do learner × oracle pairings perform? | 3 | `trac.rq3` |
| RQ4 | Can TO3 emulate a symbolic constraint checker? | 4 | `trac.rq4` |

Each run writes a CSV to `artifacts/results/`.

### Pretrained checkpoints & datasets

To reproduce the paper's **pretrained** numbers exactly, download the checkpoints
and datasets (~340 MB, not stored in Git) and place them under an *assets root*:

- **Google Drive:** https://drive.google.com/drive/folders/1PjKlSwLgHWxvHn25auu9Qx5KPHXNLYSS?usp=sharing
  (`trac_reproducibility_assets.zip`)
- Extract into `reproducibility/` (the default location), or extract anywhere and
  set `TRAC_ASSETS=/path/to/assets`.
- See [`reproducibility/README.md`](reproducibility/README.md) for the exact layout.

Without these assets the platform still runs end to end — it simply **generates**
data and **trains** fresh models on the fly.

---

## Code layout

```
TRAC/
├── app.py             # Streamlit UI (Generate → Train → Predict → Acquire → Reproduce)
├── cli.py             # headless CLI
├── requirements.txt
├── trac/
│   ├── model.py       # T-ORACLE architecture (CSPAttentionModel)
│   ├── benchmarks.py  # benchmark registry (demo/paper scales)
│   ├── datagen.py     # data generation for TO1/TO2/TO3
│   ├── train.py       # trainer + self-describing checkpoints
│   ├── checkpoint.py  # save/load model + config
│   ├── oracle.py      # TransformerOracle (neural oracle for CA)
│   ├── fastca.py      # FASTCA learner
│   ├── acquire.py     # end-to-end acquisition + evaluation
│   ├── predict.py     # classification (overall + per-arity)
│   ├── reproduce.py   # RQ1–RQ4 experiment drivers
│   └── utils.py       # paths, device, seeding, pycona bootstrap
├── pycona/            # bundled symbolic CA engine (FASTCA, QuAcq, benchmarks, …)
└── artifacts/         # generated data, trained models, results (created at runtime)
```

## Notes

- **Checkpoints are self-describing**: each `<name>.pth` is saved with a sibling
  `<name>.json` recording its dimensions, benchmark and variant, so models load
  without any manual configuration.
- **Precision vs recall of TO3**: TO3 supervises only *target* scopes, so
  acquisition recall is typically perfect while precision depends on how well the
  oracle generalizes to non-target scopes. More training epochs / larger `paper`
  datasets improve precision.
