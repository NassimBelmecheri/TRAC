"""
Unified data generation for the TRAC platform (the "generate data" stage).

A single code path produces training data for all three T-ORACLE variants,
replacing the duplicated / benchmark-specific ``data_generator.py`` scripts and
the pre-baked solution CSVs of the legacy code base.  Labels come from the
benchmark's ground-truth :class:`ConstraintOracle`, so no external data files
are required.

Variants (paper "Variants of the T-ORACLE"):
  * ``TO1`` - full assignments only (balanced solutions / non-solutions).
  * ``TO2`` - mix of full and partial assignments of varying scope sizes.
  * ``TO3`` - constraint-specific tuples restricted to each target scope
              (the best-performing variant in the paper).

Every example is encoded as a fixed length-``n`` integer vector where 0 marks an
unassigned variable (padding) and domain values are >= 1.
"""
import random

import numpy as np
import pandas as pd

from .benchmarks import build_benchmark
from pycona.utils import get_scope, check_value

VARIANTS = ("TO1", "TO2", "TO3")


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #
def _assign(X, values):
    for v, val in zip(X, values):
        v._value = int(val)


def _collect_solutions(oracle, X, limit, seed=0):
    """Collect up to ``limit`` distinct full solutions via the CP solver."""
    import cpmpy as cp

    sols = []
    model = cp.Model(oracle.constraints)

    def _cb():
        sols.append([int(v.value()) for v in X])

    try:
        model.solveAll(solution_limit=int(limit), display=_cb)
    except Exception:
        if model.solve():
            sols.append([int(v.value()) for v in X])
    return sols


def _label_visible(oracle, X, idxs):
    """Oracle label for the partial assignment on the visible variables ``idxs``
    (values must already be set on ``X``)."""
    Y = [X[i] for i in idxs]
    return int(bool(oracle.answer_membership_query(Y)))


def _make_negative_full(oracle, X, base, d_min, d_max, rng, max_tries=60):
    """Corrupt a solution until the full assignment becomes a non-solution."""
    n = len(X)
    for _ in range(max_tries):
        vals = list(base)
        k = rng.randint(1, max(1, n // 5))
        for i in rng.sample(range(n), k):
            vals[i] = rng.randint(d_min, d_max)
        _assign(X, vals)
        if not oracle.answer_membership_query(list(X)):
            return vals
    return None


# --------------------------------------------------------------------------- #
# per-variant generators
# --------------------------------------------------------------------------- #
def _gen_to1(oracle, X, d_min, d_max, n_samples, rng, seed, progress):
    """Full assignments only, balanced."""
    n = len(X)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    sols = _collect_solutions(oracle, X, limit=max(n_pos, 200), seed=seed)
    if not sols:
        raise RuntimeError("No solutions found for this benchmark.")

    rows, labels = [], []

    # Positives (sample with replacement if the pool is small, e.g. unique-solution puzzles)
    for i in range(n_pos):
        rows.append(list(sols[i % len(sols)]))
        labels.append(1)

    # Negatives
    made = 0
    guard = 0
    while made < n_neg and guard < n_neg * 200:
        guard += 1
        base = sols[rng.randrange(len(sols))]
        neg = _make_negative_full(oracle, X, base, d_min, d_max, rng)
        if neg is not None:
            rows.append(neg)
            labels.append(0)
            made += 1
            if progress and made % 250 == 0:
                progress(f"TO1 negatives {made}/{n_neg}")
    return rows, labels


def _gen_to2(oracle, X, d_min, d_max, n_samples, rng, seed, progress):
    """Mix of full and partial assignments of varying scope sizes."""
    n = len(X)
    sols = _collect_solutions(oracle, X, limit=max(n_samples // 4, 100), seed=seed)
    if not sols:
        raise RuntimeError("No solutions found for this benchmark.")

    target_pos = n_samples // 2
    target_neg = n_samples - target_pos
    rows, labels = [], []
    n_pos = n_neg = 0
    guard = 0

    while (n_pos < target_pos or n_neg < target_neg) and guard < n_samples * 300:
        guard += 1
        base = sols[rng.randrange(len(sols))]

        # visibility size: full assignment ~30% of the time, else partial
        if rng.random() < 0.3:
            k = n
        else:
            k = rng.randint(2, n)
        idxs = sorted(rng.sample(range(n), k))

        vals = [0] * n
        for i in idxs:
            vals[i] = base[i]

        # optionally corrupt some visible variables to create likely negatives
        if rng.random() < 0.5:
            for i in rng.sample(idxs, rng.randint(1, len(idxs))):
                vals[i] = rng.randint(d_min, d_max)

        _assign(X, vals)
        label = _label_visible(oracle, X, idxs)

        if label == 1 and n_pos < target_pos:
            rows.append(vals); labels.append(1); n_pos += 1
        elif label == 0 and n_neg < target_neg:
            rows.append(vals); labels.append(0); n_neg += 1

        if progress and guard % 1000 == 0:
            progress(f"TO2 pos {n_pos}/{target_pos} neg {n_neg}/{target_neg}")
    return rows, labels


def _gen_to3(oracle, X, d_min, d_max, n_samples, rng, seed, progress, theta=0.8):
    """Constraint-specific tuples restricted to each target scope."""
    n = len(X)
    var_index = {v.name: i for i, v in enumerate(X)}

    # unique scopes of the target network, each paired with the target
    # constraints acting on that scope (or a sub-scope of it). Precomputing this
    # lets us label a sampled tuple by evaluating only the few relevant
    # constraints instead of scanning the whole oracle on every sample.
    scopes = []
    scope_rel = []
    seen = set()
    cons_scopes = [(c, set(v.name for v in get_scope(c))) for c in oracle.constraints]
    for c in oracle.constraints:
        sc = get_scope(c)
        names = set(v.name for v in sc)
        key = tuple(sorted(names))
        if key not in seen and len(sc) >= 1:
            seen.add(key)
            scopes.append(sc)
            scope_rel.append([cc for cc, cs in cons_scopes if cs.issubset(names)])
    if not scopes:
        raise RuntimeError("Target network has no constraints to supervise TO3.")

    per_scope = max(50, int((n_samples / max(1, len(scopes)))))
    rows, labels = [], []

    for si, (sc, rel) in enumerate(zip(scopes, scope_rel)):
        idxs = [var_index[v.name] for v in sc]
        lb = min(int(v.get_bounds()[0]) for v in sc)
        ub = max(int(v.get_bounds()[1]) for v in sc)

        pos_bucket, neg_bucket = [], []
        target_each = per_scope // 2
        guard = 0
        while (len(pos_bucket) < target_each or len(neg_bucket) < target_each) \
                and guard < per_scope * 200:
            guard += 1
            for v in sc:
                v._value = rng.randint(lb, ub)
            label = 1 if all(check_value(c) for c in rel) else 0
            tup = [int(v.value()) for v in sc]
            if label == 1 and len(pos_bucket) < target_each:
                pos_bucket.append(tup)
            elif label == 0 and len(neg_bucket) < target_each:
                neg_bucket.append(tup)

        # apply theta: keep a proportion of positives, with an equal number of negatives
        keep = max(1, int(round(theta * max(len(pos_bucket), len(neg_bucket)))))
        rng.shuffle(pos_bucket); rng.shuffle(neg_bucket)
        for tup in pos_bucket[:keep]:
            row = [0] * n
            for i, val in zip(idxs, tup):
                row[i] = val
            rows.append(row); labels.append(1)
        for tup in neg_bucket[:keep]:
            row = [0] * n
            for i, val in zip(idxs, tup):
                row[i] = val
            rows.append(row); labels.append(0)

        if progress:
            progress(f"TO3 scope {si + 1}/{len(scopes)}")
    return rows, labels


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def generate_dataset(name, scale=None, variant="TO3", n_samples=4000,
                     seed=42, theta=0.8, progress=None):
    """Generate a labelled dataset for ``(benchmark, variant)``.

    :return: ``(DataFrame, meta)`` with columns ``var_0..var_{n-1}, label``.
             ``meta`` carries benchmark dimensions plus dataset statistics.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")

    rng = random.Random(seed)
    instance, oracle, meta = build_benchmark(name, scale)
    X = list(instance.X)
    d_min, d_max = meta["d_min"], meta["d_max"]

    if progress:
        progress(f"Building {name} [{meta['scale']}]: {meta['n_vars']} vars, "
                 f"domain {d_min}..{d_max}, {meta['num_target_constraints']} constraints")

    if variant == "TO1":
        rows, labels = _gen_to1(oracle, X, d_min, d_max, n_samples, rng, seed, progress)
    elif variant == "TO2":
        rows, labels = _gen_to2(oracle, X, d_min, d_max, n_samples, rng, seed, progress)
    else:
        rows, labels = _gen_to3(oracle, X, d_min, d_max, n_samples, rng, seed, progress, theta)

    n = meta["n_vars"]
    cols = [f"var_{i}" for i in range(n)] + ["label"]
    data = np.column_stack([np.array(rows, dtype=int), np.array(labels, dtype=int)])
    df = pd.DataFrame(data, columns=cols).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    meta = dict(meta)
    meta.update({
        "variant": variant,
        "n_samples": len(df),
        "n_positive": int((df["label"] == 1).sum()),
        "n_negative": int((df["label"] == 0).sum()),
        "seed": seed,
        "theta": theta if variant == "TO3" else None,
    })
    return df, meta
