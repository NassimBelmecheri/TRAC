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

import cpmpy as cp

from . import utils  # noqa: F401  (bootstraps pycona)
from .benchmarks import build_benchmark
from .oracle import TransformerOracle
from .fastca import FastCA

from pycona import ActiveCAEnv, PQGen, FindC, FindC2, QuAcq, MQuAcq2, GrowAcq
from pycona.answering_queries.constraint_oracle import ConstraintOracle

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
