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
import math
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


def _gen_to3(oracle, X, d_min, d_max, n_samples, rng, seed, progress,
             theta=0.8, non_constraint_ratio=1.0):
    """Constraint-specific partial assignments (paper's TO3).

    Faithful to the reference ``ScopeFocusedGenerator`` used for the paper:

      * **Stage 1 - real constraints.** For each target scope (the *right*
        scopes) we split the tuples over that scope's domain into satisfying
        (``rel(c)``) and violating (``N(c)``) tuples, and keep a proportion
        ``theta`` of each (paper: "a proportion theta of positive tuples from
        rel(c) ... together with an equal number of tuples from N(c)").
      * **Stage 2 - neutral (non-constraint) examples.** We inject partial
        assignments on variable pairs that carry *no* constraint, labelled
        valid, to teach the oracle "no constraint here => valid". Without these
        the oracle is out-of-distribution on non-target scopes and, during
        acquisition, hallucinates a constraint on every pair (e.g. learning
        120 = C(16,2) constraints on 4x4 Sudoku instead of 56). Their number is
        ``non_constraint_ratio`` times the number of Stage-1 examples.
      * **Stage 4 - balance.** The final set is balanced 50/50 valid/invalid.

    Each example is embedded into a length-``n`` row with values placed at the
    scope's variable positions and 0 elsewhere.
    """
    import itertools

    n = len(X)
    var_index = {v.name: i for i, v in enumerate(X)}

    # Unique target scopes, each paired with the target constraints acting on
    # that scope (or a sub-scope of it). These are the *right* scopes to supervise.
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

    cap_per_scope = max(20, int(n_samples / max(1, len(scopes))))
    ENUM_LIMIT = 20000
    TYPE_REP_CAP = 12   # max replication factor for rare constraint types

    # --- Stage 1: real constraint scopes (constraint-TYPE balanced) --------
    # First pass: enumerate each scope's satisfying (pos) / violating (neg)
    # tuples and derive a *type signature* -- the set of satisfying tuples,
    # which is identical for all scopes sharing the same relation (e.g. every
    # "!=" pair). Second pass: replicate the rare types (the few "=", arithmetic
    # or unary scopes) so they are not swamped by the dominant relation (e.g.
    # Zebra's 50 "!=" cliques). Without this the oracle learns "equal values =>
    # invalid" and collapses on "=" / arithmetic constraints (measured Zebra
    # per-type accuracy: != 0.90 but = 0.07, arithmetic 0.51).
    from collections import Counter
    stage1 = []   # (idxs, pos, neg, signature)
    for sc, rel in zip(scopes, scope_rel):
        idxs = [var_index[v.name] for v in sc]
        bounds = [(int(v.get_bounds()[0]), int(v.get_bounds()[1])) for v in sc]
        dom_size = 1
        for lo, hi in bounds:
            dom_size *= (hi - lo + 1)

        pos, neg = [], []
        if dom_size <= ENUM_LIMIT:
            for combo in itertools.product(*[range(lo, hi + 1) for lo, hi in bounds]):
                for v, val in zip(sc, combo):
                    v._value = int(val)
                (pos if all(check_value(c) for c in rel) else neg).append(list(combo))
            sig = frozenset(tuple(p) for p in pos)   # exact relation signature
        else:
            target = cap_per_scope * 4
            guard = 0
            while (len(pos) < target or len(neg) < target) and guard < target * 50:
                guard += 1
                combo = [rng.randint(lo, hi) for lo, hi in bounds]
                for v, val in zip(sc, combo):
                    v._value = int(val)
                (pos if all(check_value(c) for c in rel) else neg).append(list(combo))
            sig = ("sampled", len(sc), round(len(pos) / max(1, len(pos) + len(neg)), 2))
        stage1.append((idxs, pos, neg, sig))

    sig_count = Counter(sig for _, _, _, sig in stage1)
    max_sig = max(sig_count.values()) if sig_count else 1

    rows, labels = [], []
    for si, (idxs, pos, neg, sig) in enumerate(stage1):
        # keep a proportion theta of each class (independently)
        rng.shuffle(pos); rng.shuffle(neg)
        k_pos = min(len(pos), max(1, math.ceil(theta * len(pos))), cap_per_scope)
        k_neg = min(len(neg), max(1, math.ceil(theta * len(neg))), cap_per_scope)
        # replicate rare constraint types up to the frequency of the dominant one
        reps = min(TYPE_REP_CAP, max(1, round(max_sig / sig_count[sig])))
        for _ in range(reps):
            for combo in pos[:k_pos]:
                row = [0] * n
                for i, val in zip(idxs, combo):
                    row[i] = int(val)
                rows.append(row); labels.append(1)
            for combo in neg[:k_neg]:
                row = [0] * n
                for i, val in zip(idxs, combo):
                    row[i] = int(val)
                rows.append(row); labels.append(0)

        if progress and (si + 1) % 50 == 0:
            progress(f"TO3 scope {si + 1}/{len(scopes)}")

    n_base = len(rows)

    # --- Stage 1b: unary coverage (only when the target has unary constraints) ---
    # Unary constraints have very few tuples and are otherwise drowned by the
    # thousands of binary rows, so the oracle never learns to fire on a unary
    # violation - it answers "valid" for every single-variable query (observed
    # on Zebra: P(valid|violation) ~ 0.97). We therefore (i) heavily replicate
    # the real unary examples (every domain value, balanced valid/invalid) and
    # (ii) add unary NEUTRAL examples on variables carrying no unary constraint,
    # so the model learns *which* variables are unary-constrained.
    unary_scope_vars = {var_index[sc[0].name] for sc in scopes if len(sc) == 1}
    if unary_scope_vars:
        dom = list(range(d_min, d_max + 1))
        unary_reps = max(4, (3 * cap_per_scope) // max(1, len(dom)))
        # (i) boost the real unary scopes
        for sc, rel in zip(scopes, scope_rel):
            if len(sc) != 1:
                continue
            idx = var_index[sc[0].name]
            lo, hi = int(sc[0].get_bounds()[0]), int(sc[0].get_bounds()[1])
            for val in range(lo, hi + 1):
                sc[0]._value = val
                lbl = 1 if all(check_value(c) for c in rel) else 0
                for _ in range(unary_reps):
                    row = [0] * n
                    row[idx] = val
                    rows.append(row); labels.append(lbl)
        # (ii) unary neutral on variables with no unary constraint -> always valid
        free_vars = [i for i in range(n) if i not in unary_scope_vars]
        rng.shuffle(free_vars)
        per_var = max(2, unary_reps // 2)
        for i in free_vars:
            for _ in range(per_var):
                row = [0] * n
                row[i] = rng.choice(dom)
                rows.append(row); labels.append(1)
        if progress:
            progress(f"TO3 unary coverage: {len(unary_scope_vars)} constrained, "
                     f"{len(free_vars)} free vars")

    # --- Stage 2: neutral (non-constraint) examples ------------------------
    # All binary pairs that participate in a constraint (directly or as a
    # sub-pair of a larger scope) are excluded; the rest are non-constraint and
    # any assignment on them is valid. Coverage is *systematic*: every
    # non-constraint pair receives several examples (including equal values,
    # which is exactly what FASTCA probes when violating a "!=" candidate), so
    # the oracle reliably answers "valid" on non-target scopes during
    # acquisition instead of hallucinating a constraint on every pair.
    constrained_pairs = set()
    for sc in scopes:
        sidx = [var_index[v.name] for v in sc]
        for a, b in itertools.combinations(sidx, 2):
            constrained_pairs.add((min(a, b), max(a, b)))

    unconstrained_pairs = [
        (a, b) for a, b in itertools.combinations(range(n), 2)
        if (a, b) not in constrained_pairs
    ]
    rng.shuffle(unconstrained_pairs)

    made = 0
    if unconstrained_pairs and non_constraint_ratio > 0:
        n_neutral_target = int(n_base * non_constraint_ratio)
        # at least a handful of examples per non-constraint pair
        per_pair = max(4, math.ceil(n_neutral_target / len(unconstrained_pairs)))
        dom = list(range(d_min, d_max + 1))
        for (a, b) in unconstrained_pairs:
            for _ in range(per_pair):
                row = [0] * n
                # bias half the examples toward EQUAL values (the FASTCA "!=" probe)
                if rng.random() < 0.5:
                    val = rng.choice(dom)
                    row[a] = row[b] = val
                else:
                    row[a] = rng.choice(dom); row[b] = rng.choice(dom)
                rows.append(row); labels.append(1)   # non-constraint pair -> valid
                made += 1
    if progress:
        progress(f"TO3 neutral examples: {made} over {len(unconstrained_pairs)} pairs (base {n_base})")

    # --- Stage 4: gentle balance -------------------------------------------
    # Cap the majority class at MAJORITY_CAP x the minority so neither the
    # neutral positives nor the constraint negatives are starved. A strict
    # 50/50 cut would discard almost all neutral coverage on small benchmarks
    # (where the target scopes admit very few violating tuples), which is what
    # makes acquisition hallucinate a constraint on every pair.
    MAJORITY_CAP = 5
    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]
    if pos_idx and neg_idx:
        if len(pos_idx) > MAJORITY_CAP * len(neg_idx):
            pos_idx = rng.sample(pos_idx, MAJORITY_CAP * len(neg_idx))
        elif len(neg_idx) > MAJORITY_CAP * len(pos_idx):
            neg_idx = rng.sample(neg_idx, MAJORITY_CAP * len(pos_idx))
        keep = pos_idx + neg_idx
        rng.shuffle(keep)
        rows = [rows[i] for i in keep]
        labels = [labels[i] for i in keep]
    return rows, labels
# public API
# --------------------------------------------------------------------------- #
def generate_dataset(name, scale=None, variant="TO3", n_samples=4000,
                     seed=42, theta=0.8, params=None, progress=None):
    """Generate a labelled dataset for ``(benchmark, variant)``.

    :param params: optional constructor overrides for a parametric benchmark
        (e.g. custom Sudoku dimensions); forwarded to ``build_benchmark``.
    :return: ``(DataFrame, meta)`` with columns ``var_0..var_{n-1}, label``.
             ``meta`` carries benchmark dimensions plus dataset statistics.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")

    rng = random.Random(seed)
    instance, oracle, meta = build_benchmark(name, scale, params=params)
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
