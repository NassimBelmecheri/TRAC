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


def _run_learner_queries(benchmark, learner_name, scale, time_limit, oracle,
                         construct_bias=True):
    """Run ``learner_name`` on ``benchmark`` with the given ``oracle`` and return
    a DataFrame of the membership queries it generated (var columns + ``label``,
    where ``label`` is *that oracle's* answer)."""
    instance, ground, meta = build_benchmark(benchmark, scale)
    if construct_bias and len(instance.bias) == 0:
        instance.construct_bias()
    learner = _make_learner(learner_name, benchmark, time_limit)
    learner.learn(instance, oracle, verbose=0)
    buf = getattr(learner.env.metrics, "dataset_buffer", [])
    df = pd.DataFrame(buf) if buf else pd.DataFrame()
    return df, instance, ground


def _true_labels_for_queries(df, benchmark, scale):
    """Replace the ``label`` column with the *ground-truth* oracle's answer for
    each query row (used for FASTCA, whose queries were labelled by the neural
    oracle)."""
    if df.empty:
        return df
    instance, ground, meta = build_benchmark(benchmark, scale)
    X = list(instance.X)
    cols = [c for c in df.columns if c != "label"]
    vals = df[cols].values
    labels = []
    for row in vals:
        visible = []
        for v, val in zip(X, row):
            if int(val) != 0:
                v._value = int(val)
                visible.append(v)
        labels.append(1 if (visible and ground.answer_membership_query(visible)) else
                      (1 if not visible else 0))
    out = df.copy()
    out["label"] = labels
    return out


def _classify_queries(model_path, df, device):
    """Classify a query DataFrame with a neural oracle; return (acc, rec, prec, n)."""
    from .predict import classify_dataset
    if df is None or df.empty or "label" not in df.columns:
        return None
    res = classify_dataset(model_path, df, device=device)
    o = res.get("overall")
    if not o:
        return None
    return (round(100 * o["accuracy"]), round(100 * o["recall"]),
            round(100 * o["precision"]), int(o["count"]))


def _resample_scopes(query_df, benchmark, scale, samples_per_scope=50, seed=0):
    """Build a balanced per-scope test set from the *scopes* a learner queried.

    Mirrors the reference ``partial_generator.py``: take the unique variable
    scopes appearing in the learner's queries and, for each, generate a balanced
    set of valid/invalid partial assignments labelled by the ground-truth oracle.
    This gives a robust test set even when the learner (with a perfect oracle)
    converges in very few queries.
    """
    import random as _rnd
    if query_df is None or query_df.empty:
        return query_df
    instance, ground, meta = build_benchmark(benchmark, scale)
    X = list(instance.X)
    n = len(X)
    lb = min(int(v.get_bounds()[0]) for v in X)
    ub = max(int(v.get_bounds()[1]) for v in X)
    rng = _rnd.Random(seed)

    cols = [c for c in query_df.columns if c != "label"]
    scopes = set()
    for row in query_df[cols].values:
        idx = tuple(i for i, val in enumerate(row) if int(val) != 0)
        if 1 <= len(idx) <= 6:
            scopes.add(idx)
    if not scopes:
        return query_df

    rows, labels = [], []
    for idx in scopes:
        pos, neg = [], []
        guard = 0
        target = samples_per_scope // 2
        while (len(pos) < target or len(neg) < target) and guard < samples_per_scope * 40:
            guard += 1
            vis = []
            for i in idx:
                X[i]._value = rng.randint(lb, ub)
                vis.append(X[i])
            lbl = 1 if ground.answer_membership_query(vis) else 0
            vec = [0] * n
            for i in idx:
                vec[i] = int(X[i].value())
            (pos if lbl == 1 else neg).append(vec)
        for vec in pos[:target] + neg[:target]:
            rows.append(vec)
        labels += [1] * len(pos[:target]) + [0] * len(neg[:target])

    if not rows:
        return query_df
    out = pd.DataFrame(rows, columns=[f"var_{i}" for i in range(n)])
    out["label"] = labels
    return out


def rq3(benchmarks=None, learners=("FASTCA", "QuAcq", "MQuAcq2", "GrowAcq"),
        variants=("TO1", "TO2", "TO3"), scale=None, theta="0.8",
        time_limit=8, source="pretrained", epochs=120, n_samples=4000,
        save_queries=False, device=None, progress=None):
    """Reproduce Table 3 as (Accuracy, Recall, Precision) per learner x oracle.

    Methodology (paper-faithful):
      * The **classical** learners (QuAcq / MQuAcq2 / GrowAcq) are run with the
        *ground-truth* oracle to generate a realistic query distribution; the
        neural oracles (TO1/TO2/TO3) then **classify** those queries. The neural
        model is NOT used as the acquisition oracle here.
      * **FASTCA** is the exception: it is run with the *neural* oracle as the
        actual oracle (the neuro-symbolic loop); the queries it asks are then
        scored against the ground-truth answers.

    Each cell reports the oracle's (Accuracy, Recall, Precision) on that learner's
    queries.

    :param source: ``"pretrained"`` uses shipped checkpoints; ``"train"`` retrains
        oracles with the corrected data generator.
    """
    device = device or utils.get_device()
    benchmarks = benchmarks or FAST_BENCHMARKS
    classical = [lr for lr in learners if lr != "FASTCA"]
    rows = []
    from .oracle import TransformerOracle
    qdir = os.path.join(utils.RESULTS_DIR, "queries")
    if save_queries:
        os.makedirs(qdir, exist_ok=True)

    for b in benchmarks:
        # --- classical learners: run ONCE with the TRUE oracle (variant-independent)
        classical_q = {}
        for lr in classical:
            try:
                _, ground0, _ = build_benchmark(b, scale)
                dfq, _, _ = _run_learner_queries(b, lr, scale, time_limit, ground0)
                dfq = _resample_scopes(dfq, b, scale, samples_per_scope=50)
                classical_q[lr] = dfq
                if save_queries and dfq is not None and not dfq.empty:
                    dfq.to_csv(os.path.join(qdir, f"{b}_{lr}_queries.csv"), index=False)
                if progress:
                    progress(f"  {b}/{lr}: {0 if dfq is None else len(dfq)} scope-samples (true oracle)")
            except Exception as e:
                classical_q[lr] = None
                if progress:
                    progress(f"  {b}/{lr}: query-gen err {type(e).__name__}")

        # --- per oracle variant
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
                    if lr == "FASTCA":
                        # run FASTCA with the NEURAL oracle (the neuro-symbolic loop),
                        # then score its queries against the ground truth.
                        inst_f, ground_f, _ = build_benchmark(b, scale)
                        if len(inst_f.bias) == 0:
                            inst_f.construct_bias()
                        neural = TransformerOracle.from_checkpoint(mp, inst_f.X, device=device)
                        flearner = _make_learner("FASTCA", b, time_limit)
                        flearner.learn(inst_f, neural, verbose=0)
                        buf = getattr(flearner.env.metrics, "dataset_buffer", [])
                        dfq = pd.DataFrame(buf) if buf else pd.DataFrame()
                        dfq = _true_labels_for_queries(dfq, b, scale)
                        if save_queries and not dfq.empty:
                            dfq.to_csv(os.path.join(qdir, f"{b}_{v}_FASTCA_queries.csv"), index=False)
                    else:
                        dfq = classical_q.get(lr)

                    m = _classify_queries(mp, dfq, device)
                    if m is None:
                        rows.append({"benchmark": b, "oracle": v, "learner": lr,
                                     "result": "no-queries"})
                    else:
                        rows.append({"benchmark": b, "oracle": v, "learner": lr,
                                     "accuracy": m[0], "recall": m[1],
                                     "precision": m[2], "n_queries": m[3]})
                except Exception as e:
                    rows.append({"benchmark": b, "oracle": v, "learner": lr,
                                 "result": f"err:{type(e).__name__}"})
                if progress:
                    progress(f"RQ3 {b}/{v}/{lr} -> {rows[-1]}")
    return pd.DataFrame(rows)

