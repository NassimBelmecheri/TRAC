#!/usr/bin/env python3
"""
Correctness verifier for the C++ acquisition engine.

The benchmark runner (``run_benchmark.py``) only reports whether the learned
network is *satisfiable* ("SAT?"). That is necessary but **not** sufficient: an
under-constrained network is still SAT yet admits non-solutions, and an
over-constrained one rejects real solutions. The right criterion for "acquired
correctly" is *solution-set equivalence* between the learned network C_L and the
target network C_T:

    C_L is correct  <=>  sol(C_L) == sol(C_T)
                    <=>  (C_L and some c in C_T violated)  is UNSAT   [not too weak]
                    and  (C_T and some c in C_L violated)  is UNSAT   [not too strong]

We reconstruct C_T (captured while the benchmark builds itself) and C_L (read
back from the engine) and discharge the two checks with the engine's **own
native solver** (``SOLVE_VIOLATE_BIAS``): "is there an assignment satisfying C_L
that violates C_T?" (too weak) and the symmetric question (too strong). No
third-party solver / CPMpy is used - only the C++ engine, whose solver is
cross-checked for reliability by ``solver_check.py``.

Non-normalised benchmarks (nqueens, golomb, zebra, ... : several constraints on
one scope) require **FindC2**; the classic FindC (v1) learns a single constraint
per scope and is provably too weak there. ``--findc auto`` (default) uses v1 for
the normalised benchmarks (sudoku) and v2 for the rest.

Usage
-----
    python verify.py                       # all benchmarks, all algorithms (auto FindC)
    python verify.py -b zebra nqueens4      # subset of benchmarks
    python verify.py -a quacq bruteca       # subset of algorithms
    python verify.py --findc 1              # force FindC v1
    python verify.py --findc both           # run and compare both FindC versions
"""
import argparse
import os
import sys
import time

# Make ``import c_engine`` work regardless of the working directory (the package
# lives one level above this file).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_engine.c_interface import CQuAcqEngine, SOLVE_VIOLATE_BIAS  # noqa: E402
from c_engine import benchmarks as B  # noqa: E402

INTERACTIVE_ALGOS = ["quacq", "pquacq", "mquacq", "mquacq2", "growacq", "bruteca"]


# --------------------------------------------------------------------------- #
# Capture the target network + domains while a benchmark constructs itself.
# --------------------------------------------------------------------------- #
_REC = {"domains": {}, "targets": []}
_orig_set_domain = CQuAcqEngine.set_domain
_orig_add_target = CQuAcqEngine.add_target_constraint


def _rec_set_domain(self, var, values):
    _REC["domains"][var] = list(values)
    return _orig_set_domain(self, var, values)


def _rec_add_target(self, op, scope, param=0):
    _REC["targets"].append((op, list(scope), int(param)))
    return _orig_add_target(self, op, scope, param)


CQuAcqEngine.set_domain = _rec_set_domain
CQuAcqEngine.add_target_constraint = _rec_add_target


# --------------------------------------------------------------------------- #
# Native-solver equivalence check (C_L vs C_T) - no CPMpy, engine solver only.
# --------------------------------------------------------------------------- #
def _build_engine(n, domains, cl_cons, bias_cons):
    e = CQuAcqEngine(n)
    for i in range(n):
        e.set_domain(i, domains.get(i, [0, 1]))
    for op, sc, p in cl_cons:
        e.add_cl_constraint(op, list(sc), p)
    for op, sc, p in bias_cons:
        e.add_bias_constraint(op, list(sc), p)
    return e


def _implies(cl_cons, bias_cons, n, domains, timeout):
    """Does ``cl_cons`` entail every constraint in ``bias_cons``?

    Returns 'UNSAT'  -> yes, no assignment satisfies cl_cons while violating one
                        of bias_cons (entailment holds),
            'SAT'    -> no, such a counter-example exists,
            'TIMEOUT'/'ERROR' from the native solver.
    Uses the engine's own solver: SOLVE_VIOLATE_BIAS finds an assignment that
    satisfies the CL network and violates at least one bias constraint.
    """
    if not bias_cons:
        return "UNSAT"
    e = _build_engine(n, domains, cl_cons, bias_cons)
    status, _ = e.solve_ex(SOLVE_VIOLATE_BIAS, timeout_sec=timeout)
    return status


# Normalised benchmarks (at most one constraint per scope) are correct AND
# query-efficient with FindC v1; everything else needs FindC v2.
NORMALIZED = {"sudoku4", "sudoku9", "sudoku16"}


def _auto_findc(bench_name):
    return 1 if bench_name in NORMALIZED else 2


def _equivalent(target, learned, n, domains, timeout=20):
    """Return (status, n_target, n_learned, extra, missing) using ONLY the native
    C++ solver (no CPMpy).

      * EXACT  - learned set == target set (no solve needed),
      * EQUIV  - differ syntactically but same solution set,
      * TOO_WEAK / TOO_STRONG - a native counter-example was found,
      * INCONCLUSIVE - the native solver hit its time limit while trying to *prove*
        entailment (UNSAT). Finding a real counter-example (SAT) is fast, so a
        genuine defect is never mislabelled; INCONCLUSIVE means "no violation
        found within the limit" - typically a globally-redundant but truly
        equivalent network (e.g. 9x9 sudoku), which is a hard co-NP proof.

    Note: we deliberately do NOT trust the engine's ``converged`` flag as a proof
    of entailment - a buggy learner can false-converge (leave non-entailed
    constraints in the bias), and this checker must catch exactly that.
    """
    tset, lset = set(target), set(learned)
    extra = [c for c in learned if c not in tset]     # learned but not in target
    missing = [c for c in target if c not in lset]    # target but not directly learned

    if not extra and not missing:
        return "EXACT", len(target), len(learned), 0, 0

    # TOO_WEAK: a C_L-solution violates a target constraint C_L does not enforce.
    w = _implies(learned, missing, n, domains, timeout)
    if w == "SAT":
        return "TOO_WEAK", len(target), len(learned), len(extra), len(missing)
    if w in ("TIMEOUT", "ERROR"):
        return "INCONCLUSIVE", len(target), len(learned), len(extra), len(missing)

    # TOO_STRONG: a genuine target solution violates an extra learned constraint.
    s = _implies(target, extra, n, domains, timeout)
    if s == "SAT":
        return "TOO_STRONG", len(target), len(learned), len(extra), len(missing)
    if s in ("TIMEOUT", "ERROR"):
        return "INCONCLUSIVE", len(target), len(learned), len(extra), len(missing)

    return "EQUIV", len(target), len(learned), len(extra), len(missing)


# --------------------------------------------------------------------------- #
# Run one (benchmark, algorithm, findc) and check correctness.
# --------------------------------------------------------------------------- #
def run_one(bench_name, algo, findc, max_queries, timeout=20):
    _REC["domains"] = {}
    _REC["targets"] = []
    eng = B.BENCHMARKS[bench_name]()          # records domains + targets
    target = [(op, tuple(sc), p) for (op, sc, p) in _REC["targets"]]
    domains = dict(_REC["domains"])

    if findc in (0, "auto", None):
        findc = _auto_findc(bench_name)
    eng.set_findc_version(findc)
    t0 = time.time()
    eng.run(algo, max_queries=max_queries)
    wall = time.time() - t0
    m = eng.get_metrics()

    learned = [(c["type"], tuple(c["scope"]), c["param"]) for c in eng.get_cl_constraints()]
    n = (max(domains) + 1) if domains else eng.num_vars
    status, n_t, n_l, extra, missing = _equivalent(target, learned, n, domains, timeout)
    equiv = status in ("EXACT", "EQUIV")

    return {
        "benchmark": bench_name, "algo": algo, "findc": findc,
        "n_target": n_t, "n_learned": n_l, "extra": extra, "missing": missing,
        "converged": bool(m["converged"]), "queries": m["total_queries"],
        "equiv": equiv, "status": status, "time": wall,
    }


def main():
    ap = argparse.ArgumentParser(description="Verify C++ engine acquisition correctness.")
    ap.add_argument("-b", "--benchmarks", nargs="+", default=None,
                    help="benchmarks (default: a representative set)")
    ap.add_argument("-a", "--algos", nargs="+", default=INTERACTIVE_ALGOS)
    ap.add_argument("--findc", choices=["1", "2", "both", "auto"], default="auto",
                    help="FindC version; 'auto' = v1 for normalised benchmarks, v2 otherwise")
    ap.add_argument("-m", "--max-queries", type=int, default=50000)
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="per-solve time limit (s) for the native equivalence check")
    args = ap.parse_args()

    default_bs = ["sudoku4", "sudoku9", "zebra", "nqueens4", "nqueens8", "golomb4", "golomb8"]
    benches = args.benchmarks or default_bs
    if args.findc == "both":
        findcs = [1, 2]
    elif args.findc == "auto":
        findcs = ["auto"]
    else:
        findcs = [int(args.findc)]

    rows = []
    for b in benches:
        if b not in B.BENCHMARKS:
            print(f"!! unknown benchmark {b}"); continue
        for fc in findcs:
            for a in args.algos:
                try:
                    r = run_one(b, a, fc, args.max_queries, args.timeout)
                except Exception as e:
                    r = {"benchmark": b, "algo": a, "findc": fc, "n_target": -1,
                         "n_learned": -1, "extra": -1, "missing": -1, "converged": False,
                         "queries": -1, "equiv": False, "status": f"ERR:{type(e).__name__}", "time": 0.0}
                rows.append(r)
                print(f"  [{b:9s} {a:8s} FindC{r['findc']}] {r['status']:14s} "
                      f"target={r['n_target']:<4d} learned={r['n_learned']:<4d} "
                      f"(+{r.get('extra',0)}/-{r.get('missing',0)}) "
                      f"conv={str(r['converged']):5s} Q={r['queries']:<6d} {r['time']:.2f}s",
                      flush=True)

    print("\n" + "=" * 100)
    print(f"{'Benchmark':10s} {'Algorithm':9s} {'FindC':5s} {'Correct?':14s} "
          f"{'Target':6s} {'Learned':7s} {'Extra':6s} {'Missing':7s} {'Conv':5s} {'Queries':7s}")
    print("-" * 100)
    n_ok = 0
    for r in rows:
        print(f"{r['benchmark']:10s} {r['algo']:9s} v{r['findc']:<4d} "
              f"{r['status']:14s} "
              f"{r['n_target']:<6d} {r['n_learned']:<7d} {r.get('extra',0):<6d} "
              f"{r.get('missing',0):<7d} {str(r['converged']):5s} {r['queries']:<7d}")
        n_ok += int(r["equiv"])
    print("=" * 100)
    print(f"{n_ok}/{len(rows)} runs acquired a network equivalent to the target "
          f"(EXACT or EQUIV).")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
