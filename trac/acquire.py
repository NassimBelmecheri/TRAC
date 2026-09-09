"""
Constraint-network acquisition (the end-to-end "predict" stage).

Runs a symbolic CA learner - by default FASTCA (paper Algorithm 1) - driven by a
trained T-ORACLE, then evaluates the acquired network ``L`` against the
benchmark's ground-truth network ``T`` using the standard constraint-level
precision / recall (a learned constraint is "correct" iff it is logically implied
by ``T``; a target constraint is "recovered" iff it is implied by ``L``).

This unifies the fragmented ``TRAC_test.py`` experiment driver behind a single
function that sizes and loads the oracle from its self-describing checkpoint.
"""
import time
import random

import cpmpy as cp
import numpy as np

from . import utils  # noqa: F401  (bootstraps pycona)
from .benchmarks import build_benchmark
from .oracle import TransformerOracle
from .fastca import FastCA

from pycona import ActiveCAEnv, PQGen, FindC, FindC2, QuAcq, MQuAcq2, GrowAcq
from pycona.answering_queries.constraint_oracle import ConstraintOracle
from pycona.utils import check_value, get_scope

# Benchmarks whose diagonal/distance constraints benefit from FindC2.
_FINDC2 = {"nqueens", "golomb", "zebra"}

LEARNERS = ("FASTCA", "QuAcq", "MQuAcq2", "GrowAcq")


def _make_env(benchmark, time_limit):
    findc = FindC2(time_limit=time_limit) if benchmark in _FINDC2 else FindC(time_limit=time_limit)
    return ActiveCAEnv(qgen=PQGen(time_limit=time_limit), findc=findc)


def _make_learner(name, benchmark, time_limit):
    env = _make_env(benchmark, time_limit)
    if name == "FASTCA":
        return FastCA(env)
    if name == "QuAcq":
        return QuAcq(env)
    if name == "MQuAcq2":
        return MQuAcq2(env)
    if name == "GrowAcq":
        inner = _make_env(benchmark, time_limit)
        return GrowAcq(env, inner_algorithm=MQuAcq2(inner))
    raise ValueError(f"Unknown learner '{name}'. Options: {LEARNERS}")


def _implied_by(constraints, c):
    """True iff ``c`` is logically implied by ``constraints``."""
    try:
        m = cp.Model(list(constraints))
        m += ~c
        return not m.solve()
    except Exception:
        return False


def evaluate_network(learned, target):
    """Constraint-level precision / recall of ``learned`` (L) w.r.t. ``target`` (T)."""
    learned = list(learned)
    target = list(target)

    if len(learned) == 0:
        precision = 1.0 if len(target) == 0 else 0.0
    else:
        precision = sum(_implied_by(target, c) for c in learned) / len(learned)

    if len(target) == 0:
        recall = 1.0
    else:
        recall = sum(_implied_by(learned, c) for c in target) / len(target)

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "n_learned": len(learned), "n_target": len(target)}


def _rejects(constraints, X, assignment):
    """True iff the network rejects ``assignment`` (some constraint violated)."""
    for v, val in zip(X, assignment):
        v._value = int(val)
    return not all(check_value(c) for c in constraints)


def evaluate_network_semantic(learned, target, variables, n_test=2000, seed=0):
    """Solution-based (semantic) quality of ``learned`` (L) vs ``target`` (T).

    This is the standard constraint-acquisition evaluation and matches the
    paper's Table 3 metric far better than exact constraint matching: it
    compares the *accept/reject decisions* of L and T on a balanced test set,
    with **violation (reject) as the positive class**. It is invariant to
    logically-equivalent-but-syntactically-different constraints (e.g. an
    over-constrained TO1 network flags every assignment as a violation, giving
    recall 100 / precision ~0, exactly the paper's TO1 pattern).

    - Accuracy  : fraction of test assignments where L and T agree.
    - Recall    : of the assignments T rejects, the fraction L also rejects.
    - Precision : of the assignments L rejects, the fraction T also rejects.
    """
    learned = list(learned)
    target = list(target)
    X = list(variables)
    n = len(X)
    rng = random.Random(seed)

    lb = min(int(v.get_bounds()[0]) for v in X)
    ub = max(int(v.get_bounds()[1]) for v in X)

    # Collect target solutions (accepted assignments).
    sols = []
    try:
        m = cp.Model(target)
        m.solveAll(solution_limit=max(200, n_test // 4),
                   display=lambda: sols.append([int(v.value()) for v in X]))
    except Exception:
        if cp.Model(target).solve():
            sols.append([int(v.value()) for v in X])
    if not sols:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
                "n_learned": len(learned), "n_target": len(target), "n_test": 0}

    half = n_test // 2
    tests = [(sols[i % len(sols)], 0) for i in range(half)]  # solutions: T accepts (y=0)

    # Non-solutions: corrupt a solution until T rejects it (y=1).
    made, guard = 0, 0
    while made < half and guard < half * 200:
        guard += 1
        asg = list(sols[rng.randrange(len(sols))])
        for _ in range(rng.randint(1, 3)):
            asg[rng.randrange(n)] = rng.randint(lb, ub)
        if _rejects(target, X, asg):
            tests.append((asg, 1)); made += 1

    tp = fp = tn = fn = 0
    for asg, y_true in tests:
        y_pred = 1 if _rejects(learned, X, asg) else 0
        if y_true == 1 and y_pred == 1: tp += 1
        elif y_true == 0 and y_pred == 1: fp += 1
        elif y_true == 0 and y_pred == 0: tn += 1
        else: fn += 1

    total = max(1, tp + fp + tn + fn)
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if tp else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    accuracy = (tp + tn) / total
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "accuracy": round(accuracy, 4),
            "n_learned": len(learned), "n_target": len(target), "n_test": len(tests)}


def evaluate_network_exact(learned, target):
    """Exact (logical-equivalence) constraint match - the reference/paper metric.

    A learned constraint counts as correct iff it is logically **equivalent** to
    some target constraint (each implies the other); a target constraint is
    recovered iff some learned constraint is equivalent to it. This is robust to
    syntactic ordering (``X != Y`` vs ``Y != X``) yet, unlike the implication
    metric, does not credit merely-implied constraints.
    """
    learned = list(learned)
    target = list(target)

    def equiv(c, t):
        return _implied_by([t], c) and _implied_by([c], t)

    if len(learned) == 0:
        precision = 1.0 if len(target) == 0 else 0.0
    else:
        precision = sum(any(equiv(c, t) for t in target) for c in learned) / len(learned)
    if len(target) == 0:
        recall = 1.0
    else:
        recall = sum(any(equiv(c, t) for c in learned) for t in target) / len(target)

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "n_learned": len(learned), "n_target": len(target)}


def acquire(benchmark, scale=None, model_path=None, learner="FASTCA",
            time_limit=10, device=None, verbose=0, progress=None):
    """Run end-to-end acquisition with a trained neural oracle.

    :return: dict with learned constraints (as strings), evaluation and query stats.
    """
    device = device or utils.get_device()
    if progress:
        progress(f"Building benchmark {benchmark} [{scale or 'default'}] ...")
    instance, ground_oracle, meta = build_benchmark(benchmark, scale)

    if progress:
        progress("Loading neural oracle checkpoint ...")
    oracle = TransformerOracle.from_checkpoint(model_path, instance.X, device=device)

    if progress:
        progress("Constructing bias ...")
    if len(instance.bias) == 0:
        instance.construct_bias()
    bias_size = len(instance.bias)

    learner_obj = _make_learner(learner, benchmark, time_limit)

    if progress:
        progress(f"Running {learner} with the neural oracle ...")
    t0 = time.time()
    learner_obj.learn(instance, oracle, verbose=verbose)
    elapsed = time.time() - t0

    env = learner_obj.env
    try:
        env.metrics.finalize_statistics()
    except Exception:
        pass

    learned = list(env.instance.cl)
    if progress:
        progress("Evaluating acquired network vs ground truth ...")
    ev = evaluate_network(learned, ground_oracle.constraints)

    return {
        "benchmark": benchmark,
        "scale": meta["scale"],
        "learner": learner,
        "n_vars": meta["n_vars"],
        "bias_size": bias_size,
        "n_target": len(ground_oracle.constraints),
        "n_learned": len(learned),
        "membership_queries": int(getattr(env.metrics, "membership_queries_count", 0)),
        "total_queries": int(getattr(env.metrics, "total_queries", 0)),
        "wall_time_s": round(elapsed, 2),
        "evaluation": ev,
        "learned_constraints": [str(c) for c in learned],
    }
