#!/usr/bin/env python3
"""
Reliability check for the native C++ solver (`FastSolver`, reached through
`CQuAcqEngine.solve_ex`).

It cross-checks the engine's `SOLVE_ANY` and `SOLVE_VIOLATE_BIAS` verdicts against
an **independent brute-force ground truth** (pure-Python enumeration - no CPMpy)
on hundreds of small random constraint networks, plus every benchmark's target
network. For each instance it verifies:

  * the SAT/UNSAT verdict matches brute force,
  * any returned assignment really satisfies the constraints (SAT is not a lie),
  * `SOLVE_VIOLATE_BIAS` returns an assignment that satisfies CL and violates at
    least one bias constraint.

Run:  python solver_check.py            (default 400 random + benchmarks)
      python solver_check.py -n 2000
"""
import argparse
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_engine.c_interface import (  # noqa: E402
    CQuAcqEngine, SOLVE_ANY, SOLVE_VIOLATE_BIAS,
    OP_NE, OP_EQ, OP_GT, OP_LT, OP_GE, OP_LE,
    OP_DIFF_OFFSET_EQ, OP_DIFF_OFFSET_NE,
    OP_ABS_DIFF_EQ, OP_ABS_DIFF_NE, OP_ABS_DIFF_LT, OP_ABS_DIFF_LE,
    OP_ABS_DIFF_GT, OP_ABS_DIFF_GE, OP_SUM_OFFSET_EQ, OP_SUM_OFFSET_NE,
    OP_GOLOMB_EQ, OP_GOLOMB_NE, OP_GOLOMB_LT, OP_GOLOMB_GT,
    OP_TERNARY_DIFF_EQ, OP_TERNARY_SUM_EQ,
    OP_UNARY_EQ, OP_UNARY_NE, OP_UNARY_GT, OP_UNARY_LT, OP_UNARY_GE, OP_UNARY_LE,
)
from c_engine import benchmarks as B  # noqa: E402


# --- pure-Python constraint semantics (must mirror c_engine/constraint.h) --- #
def sat(op, sc, p, a):
    x = [a[i] for i in sc]
    if op == OP_NE: return x[0] != x[1]
    if op == OP_EQ: return x[0] == x[1]
    if op == OP_GT: return x[0] > x[1]
    if op == OP_LT: return x[0] < x[1]
    if op == OP_GE: return x[0] >= x[1]
    if op == OP_LE: return x[0] <= x[1]
    if op == OP_DIFF_OFFSET_EQ: return (x[0] - x[1]) == p
    if op == OP_DIFF_OFFSET_NE: return (x[0] - x[1]) != p
    if op == OP_ABS_DIFF_EQ: return abs(x[0] - x[1]) == p
    if op == OP_ABS_DIFF_NE: return abs(x[0] - x[1]) != p
    if op == OP_ABS_DIFF_LT: return abs(x[0] - x[1]) < p
    if op == OP_ABS_DIFF_LE: return abs(x[0] - x[1]) <= p
    if op == OP_ABS_DIFF_GT: return abs(x[0] - x[1]) > p
    if op == OP_ABS_DIFF_GE: return abs(x[0] - x[1]) >= p
    if op == OP_SUM_OFFSET_EQ: return (x[0] + x[1]) == p
    if op == OP_SUM_OFFSET_NE: return (x[0] + x[1]) != p
    if op == OP_GOLOMB_EQ: return abs(x[0] - x[1]) == abs(x[2] - x[3])
    if op == OP_GOLOMB_NE: return abs(x[0] - x[1]) != abs(x[2] - x[3])
    if op == OP_GOLOMB_LT: return abs(x[0] - x[1]) < abs(x[2] - x[3])
    if op == OP_GOLOMB_GT: return abs(x[0] - x[1]) > abs(x[2] - x[3])
    if op == OP_TERNARY_DIFF_EQ: return (x[0] - x[1]) == x[2]
    if op == OP_TERNARY_SUM_EQ: return (x[0] + x[1]) == x[2]
    if op == OP_UNARY_EQ: return x[0] == p
    if op == OP_UNARY_NE: return x[0] != p
    if op == OP_UNARY_GT: return x[0] > p
    if op == OP_UNARY_LT: return x[0] < p
    if op == OP_UNARY_GE: return x[0] >= p
    if op == OP_UNARY_LE: return x[0] <= p
    raise ValueError(op)


def brute_any(cons, domains, n):
    """Ground truth: return a satisfying assignment or None (UNSAT)."""
    for combo in itertools.product(*[domains[i] for i in range(n)]):
        if all(sat(op, sc, p, combo) for op, sc, p in cons):
            return list(combo)
    return None


def brute_violate(cl, bias, domains, n):
    """Ground truth for SOLVE_VIOLATE_BIAS: assignment satisfying CL and
    violating >=1 bias constraint, or None."""
    for combo in itertools.product(*[domains[i] for i in range(n)]):
        if all(sat(op, sc, p, combo) for op, sc, p in cl):
            if any(not sat(op, sc, p, combo) for op, sc, p in bias):
                return list(combo)
    return None


# --------------------------------------------------------------------------- #
BIN_OPS = [(OP_NE, 0), (OP_EQ, 0), (OP_GT, 0), (OP_LT, 0), (OP_GE, 0), (OP_LE, 0),
           (OP_DIFF_OFFSET_EQ, 1), (OP_DIFF_OFFSET_NE, 1),
           (OP_ABS_DIFF_EQ, 1), (OP_ABS_DIFF_NE, 2), (OP_SUM_OFFSET_EQ, 4)]
UN_OPS = [(OP_UNARY_EQ, 1), (OP_UNARY_NE, 2), (OP_UNARY_GT, 1), (OP_UNARY_LE, 2)]


def _mk_engine(n, domains, cl, bias):
    e = CQuAcqEngine(n)
    for i in range(n):
        e.set_domain(i, domains[i])
    for op, sc, p in cl:
        e.add_cl_constraint(op, list(sc), p)
    for op, sc, p in bias:
        e.add_bias_constraint(op, list(sc), p)
    return e


def random_case(rng):
    n = rng.randint(2, 4)
    lo = 1
    hi = rng.randint(2, 5)
    domains = {i: list(range(lo, hi + 1)) for i in range(n)}
    cl, bias = [], []
    for _ in range(rng.randint(1, 5)):
        if n >= 2 and rng.random() < 0.75:
            i, j = rng.sample(range(n), 2)
            op, p = rng.choice(BIN_OPS)
            (cl if rng.random() < 0.6 else bias).append((op, (i, j), p))
        else:
            i = rng.randrange(n)
            op, p = rng.choice(UN_OPS)
            (cl if rng.random() < 0.6 else bias).append((op, (i,), p))
    return n, domains, cl, bias


def run_random(n_cases, seed, verbose):
    rng = random.Random(seed)
    fails = []
    n_any = n_viol = 0
    for k in range(n_cases):
        n, domains, cl, bias = random_case(rng)

        # --- SOLVE_ANY vs brute force (use CL only) --------------------------
        truth = brute_any(cl, domains, n)
        eng = _mk_engine(n, domains, cl, [])
        status, asg = eng.solve_ex(SOLVE_ANY, timeout_sec=5.0)
        n_any += 1
        if status == "TIMEOUT":
            fails.append(("ANY/timeout", n, domains, cl, [], truth, status, asg))
        elif (truth is None) != (status == "UNSAT"):
            fails.append(("ANY/verdict", n, domains, cl, [], truth, status, asg))
        elif status == "SAT":
            a = asg[:n]
            if not all(sat(op, sc, p, a) for op, sc, p in cl):
                fails.append(("ANY/invalid-sol", n, domains, cl, [], truth, status, a))

        # --- SOLVE_VIOLATE_BIAS vs brute force -------------------------------
        if bias:
            vtruth = brute_violate(cl, bias, domains, n)
            eng2 = _mk_engine(n, domains, cl, bias)
            st2, a2 = eng2.solve_ex(SOLVE_VIOLATE_BIAS, timeout_sec=5.0)
            n_viol += 1
            if st2 == "TIMEOUT":
                fails.append(("VIOL/timeout", n, domains, cl, bias, vtruth, st2, a2))
            elif (vtruth is None) != (st2 == "UNSAT"):
                fails.append(("VIOL/verdict", n, domains, cl, bias, vtruth, st2, a2))
            elif st2 == "SAT":
                a = a2[:n]
                ok_cl = all(sat(op, sc, p, a) for op, sc, p in cl)
                ok_v = any(not sat(op, sc, p, a) for op, sc, p in bias)
                if not (ok_cl and ok_v):
                    fails.append(("VIOL/invalid", n, domains, cl, bias, vtruth, st2, a))

    print(f"random: {n_any} SOLVE_ANY + {n_viol} SOLVE_VIOLATE_BIAS checks, "
          f"{len(fails)} mismatch(es)")
    for f in fails[:12 if not verbose else len(fails)]:
        print("  MISMATCH", f[0], "n=", f[1], "dom=", f[2], "cl=", f[3],
              "bias=", f[4], "brute=", f[5], "engine=", f[6], f[7])
    return len(fails)


def run_benchmarks():
    """Solve each benchmark's target network and verify the solution is valid."""
    rec = {"dom": {}, "tgt": []}
    _sd, _at = CQuAcqEngine.set_domain, CQuAcqEngine.add_target_constraint
    CQuAcqEngine.set_domain = lambda s, v, vals, _o=_sd: (rec["dom"].__setitem__(v, list(vals)), _o(s, v, vals))[1]
    CQuAcqEngine.add_target_constraint = lambda s, op, sc, p=0, _o=_at: (rec["tgt"].append((op, tuple(sc), p)), _o(s, op, sc, p))[1]

    fails = 0
    for name in ["sudoku4", "sudoku9", "zebra", "nqueens4", "nqueens8", "golomb4"]:
        rec["dom"], rec["tgt"] = {}, []
        B.BENCHMARKS[name]()
        tgt, dom = list(rec["tgt"]), dict(rec["dom"])
        n = max(dom) + 1
        e = CQuAcqEngine(n)
        for i in range(n):
            e.set_domain(i, dom[i])
        for op, sc, p in tgt:
            e.add_cl_constraint(op, list(sc), p)
        status, asg = e.solve_ex(SOLVE_ANY, timeout_sec=30.0)
        ok = status == "SAT" and all(sat(op, sc, p, asg[:n]) for op, sc, p in tgt)
        print(f"  {name:9s}: target solve -> {status:8s} "
              f"{'valid solution' if ok else 'NOT VERIFIED'}  ({len(tgt)} constraints)")
        if not ok:
            fails += 1

    CQuAcqEngine.set_domain, CQuAcqEngine.add_target_constraint = _sd, _at
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--n-cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    print("=== native solver reliability check ===")
    print("\n[1] benchmark target networks (should all be SAT with a valid solution)")
    f_bench = run_benchmarks()
    print(f"\n[2] {args.n_cases} random small networks vs brute-force ground truth")
    f_rand = run_random(args.n_cases, args.seed, args.verbose)

    total = f_bench + f_rand
    print("\n" + "=" * 60)
    print("RESULT:", "RELIABLE - all checks passed" if total == 0
          else f"UNRELIABLE - {total} mismatch(es) found")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
