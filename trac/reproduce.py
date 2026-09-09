"""
Reproduction of the paper's experiments (RQ1-RQ4).

Where possible this reuses the assets already shipped in the repository - the
base datasets in ``conacq_datasets/``, the per-variant datasets in ``TO2/`` and
``TO3/`` and the 98 pretrained ``.pth`` checkpoints in ``TO{1,2,3}/models/`` -
so the paper's tables can be regenerated without a cluster.

    RQ1  FASTCA vs QuAcq / MQuAcq2 / GrowAcq  -> query count & timing (Table 1)
    RQ2  TO1 / TO2 / TO3 classification       -> Acc / Rec / Prec    (Table 2)
    RQ3  learner x oracle acquisition         -> Prec / Rec / F1     (Table 3)
    RQ4  TO3 as a symbolic constraint checker -> Acc / Rec / Prec    (Table 4)

Legacy naming is reconciled here too: the same benchmark is called e.g.
``latin_squares_10`` in TO1/TO3 but ``latin_10`` in TO2, and ``exam_timetabling``
vs ``exam`` - see :data:`LEGACY_TOKENS`.
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from . import utils
from .checkpoint import load_checkpoint
from .predict import _predict_proba
from .datagen import generate_dataset
from .train import train_oracle
from .benchmarks import build_benchmark
from .acquire import (_make_learner, evaluate_network, evaluate_network_semantic,
                      evaluate_network_exact)

REPO = utils.REPO_ROOT


def _default_assets_root():
    """Root under which the pretrained checkpoints and per-variant datasets live.

    Expected layout::

        <assets>/conacq_datasets/         # TO1 datasets
        <assets>/TO1/models/              # TO1 checkpoints
        <assets>/TO2/  and  TO2/models/   # TO2 datasets + checkpoints
        <assets>/TO3/  and  TO3/models/   # TO3 datasets + checkpoints

    Resolution order:
      1. the ``TRAC_ASSETS`` environment variable, if set;
      2. the original repository tree (``REPO``) when it still holds the assets;
      3. ``<repo>/reproducibility`` (created on demand) otherwise.
    """
    env = os.environ.get("TRAC_ASSETS")
    if env:
        return env
    if os.path.isdir(os.path.join(REPO, "TO3")) or \
       os.path.isdir(os.path.join(REPO, "conacq_datasets")):
        return REPO
    return os.path.join(utils.PLATFORM_ROOT, "reproducibility")


# Directory holding the pretrained checkpoints / datasets used for reproduction.
ASSETS_ROOT = _default_assets_root()

# benchmark key -> (TO1/conacq token, TO2 token, TO3 token)
LEGACY_TOKENS = {
    "sudoku":           ("sudoku_9", "sudoku_9", "sudoku_9"),
    "jsudoku":          ("jsudoku", "jsudoku", "jsudoku"),
    "latin_squares":    ("latin_squares_10", "latin_10", "latin_squares_10"),
    "nqueens":          ("nqueens_8", "nqueens_8", "nqueens_8"),
    "job_shop":         ("job_shop", "job_shop", "job_shop"),
    "murder":           ("murder", "murder", "murder"),
    "exam_timetabling": ("exam_timetabling", "exam", "exam_timetabling"),
    "nurse_rostering":  ("nurse_rostering", "nurse", "nurse_rostering"),
    "zebra":            ("zebra", "zebra", "zebra"),
    "golomb":           ("golomb_8", "golomb_8", "golomb_8"),
    "random122":        ("random122", "random122", "random122"),
    "random495":        ("random495", "random495", "random495"),
}

# The paper's Table 2 benchmark set (nqueens and golomb are extra benchmarks
# present in the pycona code but NOT evaluated in the paper).
PAPER_BENCHMARKS = ["sudoku", "jsudoku", "latin_squares", "job_shop", "murder",
                    "zebra", "random122", "random495", "exam_timetabling",
                    "nurse_rostering"]

# Small / fast subset used as a sensible default for the heavier RQs.
FAST_BENCHMARKS = ["murder", "zebra", "random122"]


# --------------------------------------------------------------------------- #
# legacy asset locators
# --------------------------------------------------------------------------- #
def find_dataset(benchmark, variant, theta="0.8"):
    """Return ``(path, source)`` for an existing dataset, or ``(None, None)``."""
    t1, t2, t3 = LEGACY_TOKENS[benchmark]
    if variant == "TO1":
        p = os.path.join(ASSETS_ROOT, "conacq_datasets", f"{t1}.csv")
    elif variant == "TO2":
        p = os.path.join(ASSETS_ROOT, "TO2", f"dataset_{t2}.csv")
    else:
        p = os.path.join(ASSETS_ROOT, "TO3", f"dataset_{t3}_{theta}.csv")
    return (p, "existing") if os.path.exists(p) else (None, None)


def find_model(benchmark, variant, theta="0.8"):
    """Return the path of an existing pretrained checkpoint, or ``None``."""
    hits = find_models(benchmark, variant, theta)
    return hits[0] if hits else None


def find_models(benchmark, variant, theta="0.8"):
    """Return all matching pretrained checkpoints (a benchmark may ship several,
    e.g. nqueens has 4/6/8-variable models)."""
    t1, t2, t3 = LEGACY_TOKENS[benchmark]
    if variant == "TO1":
        pat = os.path.join(ASSETS_ROOT, "TO1", "models", f"best_model_{t1}.csv_*_0.0.pth")
    elif variant == "TO2":
        pat = os.path.join(ASSETS_ROOT, "TO2", "models", f"best_model_dataset_{t2}.csv_*_0.0.pth")
    else:
        pat = os.path.join(ASSETS_ROOT, "TO3", "models", f"best_model_dataset_{t3}_{theta}.csv_*_0.0.pth")
    return sorted(glob.glob(pat))


# --------------------------------------------------------------------------- #
# RQ2 - oracle classification (Table 2)
# --------------------------------------------------------------------------- #
def _metrics(y, pred):
    return (round(100 * accuracy_score(y, pred)),
            round(100 * recall_score(y, pred, zero_division=0)),
            round(100 * precision_score(y, pred, zero_division=0)))


def _eval_pretrained(model_path, df, device, holdout=True, seed=42):
    model, cfg = load_checkpoint(model_path, device=device)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(int)
    if X.shape[1] != cfg["num_positions"]:
        return None
    if holdout and len(set(y.tolist())) > 1 and len(y) > 20:
        _, X, _, y = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    # legacy convention: ignore all-zero rows
    keep = np.any(X != 0, axis=1)
    X, y = X[keep], y[keep]
    X = np.clip(X, 0, cfg["num_values"])  # guard against out-of-vocab values
    probs = _predict_proba(model, X, device)
    pred = (probs >= 0.5).astype(int)
    return _metrics(y, pred)


def rq2(benchmarks=None, variants=("TO1", "TO2", "TO3"), source="pretrained",
        theta="0.8", epochs=60, generate_missing=False, device=None, progress=None):
    """Reproduce Table 2 (oracle classification per benchmark / variant).

    By default only the datasets shipped in the repository are used; TO3 cells
    with no shipped dataset are reported as ``no-ds``. Pass
    ``generate_missing=True`` to synthesise the missing (paper-scale) test data
    instead - correct but slow for the large benchmarks.
    """
    device = device or utils.get_device()
    benchmarks = benchmarks or PAPER_BENCHMARKS
    rows = []
    for b in benchmarks:
        row = {"benchmark": b}
        for v in variants:
            ds_path, _ = find_dataset(b, v, theta)
            try:
                if ds_path is not None:
                    df = pd.read_csv(ds_path)
                elif generate_missing:  # no shipped dataset: synthesise (paper scale)
                    df, _ = generate_dataset(b, scale="paper", variant=v,
                                             n_samples=3000, seed=42)
                else:
                    row[v] = "no-ds"
                    if progress:
                        progress(f"RQ2 {b}/{v} -> no-ds")
                    continue

                if source == "pretrained":
                    mps = find_models(b, v, theta)
                    m = None
                    for mp in mps:  # try each size until dims match
                        m = _eval_pretrained(mp, df, device)
                        if m is not None:
                            break
                    if not mps:
                        row[v] = "no-model"
                        if progress:
                            progress(f"RQ2 {b}/{v} -> no-model")
                        continue
                else:
                    m = _train_and_eval(df, b, v, epochs, device)

                row[v] = f"{m[0]}/{m[1]}/{m[2]}" if m else "-"
            except Exception as e:
                row[v] = f"err:{type(e).__name__}"
            if progress:
                progress(f"RQ2 {b}/{v} -> {row.get(v)}")
        rows.append(row)
    return pd.DataFrame(rows)


def _train_and_eval(df, benchmark, variant, epochs, device):
    res = train_oracle(df, benchmark=benchmark, variant=variant,
                       model_name=f"rq2_{benchmark}_{variant}", epochs=epochs,
                       device=device)
    m = res["best_metrics"]
    return (round(100 * m["accuracy"]), round(100 * m["recall"]),
            round(100 * m["precision"]))


# --------------------------------------------------------------------------- #
# RQ4 - TO3 as a constraint checker (Table 4)
# --------------------------------------------------------------------------- #
CHECKER_CONSTRAINTS = [
    ("Binary",     "X>Y",           2, lambda v: v[0] > v[1]),
    ("Binary",     "X!=Y",          2, lambda v: v[0] != v[1]),
    ("Ternary",    "X+Y>Z",         3, lambda v: v[0] + v[1] > v[2]),
    ("Ternary",    "X+Y!=Z",        3, lambda v: v[0] + v[1] != v[2]),
    ("Ternary",    "|X-Y|>|X-Z|",   3, lambda v: abs(v[0] - v[1]) > abs(v[0] - v[2])),
    ("Ternary",    "|X+Y|!=|X-Z|",  3, lambda v: abs(v[0] + v[1]) != abs(v[0] - v[2])),
    ("Quaternary", "X+Y>Z+T",       4, lambda v: v[0] + v[1] > v[2] + v[3]),
    ("Quaternary", "X+Y!=Z+T",      4, lambda v: v[0] + v[1] != v[2] + v[3]),
    ("Quaternary", "|X-Y|>|Z-T|",   4, lambda v: abs(v[0] - v[1]) > abs(v[2] - v[3])),
    ("Quaternary", "|X+Y|!=|Z-T|",  4, lambda v: abs(v[0] + v[1]) != abs(v[2] - v[3])),
]


def _checker_dataset(arity, pred, domain=20, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = [], []
    target = n // 2
    guard = 0
    while (len(pos) < target or len(neg) < target) and guard < n * 40:
        guard += 1
        v = rng.integers(1, domain + 1, size=arity).tolist()
        if pred(v):
            if len(pos) < target:
                pos.append(v + [1])
        elif len(neg) < target:
            neg.append(v + [0])
    data = np.array(pos + neg, dtype=int)
    cols = [f"var_{i}" for i in range(arity)] + ["label"]
    return pd.DataFrame(data, columns=cols).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def rq4(constraints=None, domain=30, n_samples=5000, epochs=50, device=None, progress=None):
    """Reproduce Table 4 (TO3 emulating a symbolic constraint checker)."""
    device = device or utils.get_device()
    picks = constraints or [c[1] for c in CHECKER_CONSTRAINTS]
    rows = []
    for arity_name, name, arity, pred in CHECKER_CONSTRAINTS:
        if name not in picks:
            continue
        df = _checker_dataset(arity, pred, domain=domain, n=n_samples)
        res = train_oracle(df, benchmark=f"checker_{name}", variant="TO3",
                           model_name=f"rq4_{name}", epochs=epochs,
                           batch_size=128, device=device)
        m = res["best_metrics"]
        rows.append({
            "arity": arity_name, "constraint": name,
            "accuracy": round(100 * m["accuracy"], 1),
            "recall": round(100 * m["recall"], 1),
            "precision": round(100 * m["precision"], 1),
        })
        if progress:
            progress(f"RQ4 {name} -> acc {rows[-1]['accuracy']}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# RQ1 - CA-learner efficiency with the real oracle (Table 1)
# --------------------------------------------------------------------------- #
def rq1(benchmarks=None, learners=("FASTCA", "QuAcq", "MQuAcq2", "GrowAcq"),
        scale=None, time_limit=10, progress=None):
    """Reproduce Table 1 (learner query count / timing, ground-truth oracle)."""
    benchmarks = benchmarks or FAST_BENCHMARKS
    rows = []
    for b in benchmarks:
        for lr in learners:
            try:
                instance, oracle, meta = build_benchmark(b, scale)
                if len(instance.bias) == 0:
                    instance.construct_bias()
                learner_obj = _make_learner(lr, b, time_limit)
                t0 = time.time()
                learner_obj.learn(instance, oracle, verbose=0)
                elapsed = time.time() - t0
                env = learner_obj.env
                try:
                    env.metrics.finalize_statistics()
                except Exception:
                    pass
                q = int(getattr(env.metrics, "membership_queries_count", 0)) or \
                    int(getattr(env.metrics, "total_queries", 0))
                rows.append({
                    "benchmark": b, "learner": lr,
                    "queries": q,
                    "avg_query_ms": round(1000 * elapsed / q, 2) if q else 0.0,
                    "total_time_s": round(elapsed, 2),
                    "learned": len(env.instance.cl),
                })
            except Exception as e:
                rows.append({"benchmark": b, "learner": lr, "queries": None,
                             "avg_query_ms": None, "total_time_s": None,
                             "learned": f"err:{type(e).__name__}"})
            if progress:
                progress(f"RQ1 {b}/{lr} -> {rows[-1]}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# RQ3 - learner x neural-oracle acquisition (Table 3)
# --------------------------------------------------------------------------- #
def _get_or_train_oracle(b, v, scale, device, epochs, n_samples, theta, progress):
    """Return the path of a model for ``(benchmark, variant)``, training a fresh
    one with the (corrected) data generator when ``epochs`` is given."""
    from .oracle import TransformerOracle  # noqa
    df, meta = generate_dataset(b, scale=scale, variant=v, n_samples=n_samples,
                                seed=42, theta=float(theta))
    res = train_oracle(df, benchmark=b, scale=meta["scale"], variant=v,
                       model_name=f"rq3_{b}_{v}", epochs=epochs, batch_size=128,
                       device=device)
    if progress:
        m = res["best_metrics"]
        progress(f"  trained {b}/{v}: acc={m['accuracy']:.3f} rec={m['recall']:.3f}")
    return res["model_path"]


def rq3(benchmarks=None, learners=("FASTCA", "QuAcq", "MQuAcq2", "GrowAcq"),
        variants=("TO1", "TO2", "TO3"), scale=None, theta="0.8",
        time_limit=8, source="pretrained", epochs=120, n_samples=4000,
        metric="exact", device=None, progress=None):
    """Reproduce Table 3 (acquired-network quality for learner x oracle pairs).

    :param source: ``"pretrained"`` uses the shipped legacy checkpoints;
        ``"train"`` trains fresh oracles with the corrected data generator
        (needed to reproduce the paper's TO3 acquisition numbers, since the
        shipped checkpoints predate the data-generation fix).
    :param metric: ``"exact"`` (default, logical-equivalence match - the paper's
        metric), ``"implication"`` (learned implied by target), or ``"semantic"``
        (solution-based; note it is very strict on unique-solution puzzles).
    """
    device = device or utils.get_device()
    benchmarks = benchmarks or FAST_BENCHMARKS
    rows = []
    from .oracle import TransformerOracle
    for b in benchmarks:
        for v in variants:
            try:
                if source == "train":
                    mp = _get_or_train_oracle(b, v, scale, device, epochs,
                                              n_samples, theta, progress)
                else:
                    mp = find_model(b, v, theta)
                    if mp is None:
                        continue
            except Exception as e:
                rows.append({"benchmark": b, "oracle": v, "learner": "*",
                             "result": f"train-err:{type(e).__name__}"})
                continue

            for lr in learners:
                try:
                    instance, ground, meta = build_benchmark(b, scale)
                    if instance.X and load_checkpoint(mp, device=device)[1]["num_positions"] != len(instance.X):
                        rows.append({"benchmark": b, "oracle": v, "learner": lr,
                                     "result": "dim-mismatch"})
                        continue
                    oracle = TransformerOracle.from_checkpoint(mp, instance.X, device=device)
                    if len(instance.bias) == 0:
                        instance.construct_bias()
                    learner_obj = _make_learner(lr, b, time_limit)
                    learner_obj.learn(instance, oracle, verbose=0)
                    learned = learner_obj.env.instance.cl
                    if metric == "exact":
                        ev = evaluate_network_exact(learned, ground.constraints)
                    elif metric == "semantic":
                        ev = evaluate_network_semantic(learned, ground.constraints, instance.X)
                    else:
                        ev = evaluate_network(learned, ground.constraints)
                    rows.append({"benchmark": b, "oracle": v, "learner": lr,
                                 "accuracy": ev.get("accuracy"),
                                 "precision": ev["precision"], "recall": ev["recall"],
                                 "f1": ev["f1"], "learned": ev["n_learned"],
                                 "target": ev["n_target"]})
                except Exception as e:
                    rows.append({"benchmark": b, "oracle": v, "learner": lr,
                                 "result": f"err:{type(e).__name__}"})
                if progress:
                    progress(f"RQ3 {b}/{v}/{lr} -> {rows[-1]}")
    return pd.DataFrame(rows)

