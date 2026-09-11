#!/usr/bin/env python3
import sys
import os
import time
import argparse

sys.path.insert(0, os.path.abspath("."))

from c_engine.benchmarks import BENCHMARKS
from c_engine import OP_NE

def run_single_algo(benchmark_name, algo_name, constructor_fn, max_queries, use_findc2=False):
    eng = constructor_fn()
    if use_findc2:
        eng.set_findc_version(2)
        
    t0 = time.time()
    learned_active = eng.run(algo_name, max_queries=max_queries)
    t1 = time.time()
    
    m = eng.get_metrics()
    rem_bias = eng.get_bias_count()
    
    sol = eng.solve()
    is_sat = sol is not None

    if m["converged"]:
        eng.promote_bias_to_cl()
    total_cl = eng.get_cl_count()
    
    return {
        "algo": algo_name.upper(),
        "active": learned_active,
        "implied": rem_bias,
        "total": total_cl,
        "is_sat": is_sat,
        "m_queries": m["membership_queries"],
        "fs_queries": m["findscope_queries"],
        "fc_queries": m["findc_queries"],
        "total_queries": m["total_queries"],
        "cpp_ms": m["total_time_sec"] * 1000.0,
        "wall_ms": (t1 - t0) * 1000.0,
        "sol": sol,
        "engine": eng
    }

def main():
    parser = argparse.ArgumentParser(description="High-Speed Constraint Acquisition Benchmark Suite (C++ Engine)")
    parser.add_argument("benchmark", choices=list(BENCHMARKS.keys()), help="Benchmark name (e.g., zebra, sudoku16, sudoku9, nqueens)")
    parser.add_argument("--algo", "-a", default="all", choices=["all", "quacq", "pquacq", "growacq", "bruteca"], help="Algorithm to run")
    parser.add_argument("--max-queries", "-m", type=int, default=10000, help="Maximum query budget")
    parser.add_argument("--findc2", action="store_true", help="Use FindC2 for multi-constraint scopes")
    parser.add_argument("-n", type=int, default=8, help="Size parameter n (for nqueens)")
    
    args = parser.parse_args()
    
    b_name = args.benchmark
    constructor_fn = lambda: BENCHMARKS[b_name](n=args.n)
    
    # Instance probe
    probe_eng = constructor_fn()
    print("=" * 80)
    print(f"  BENCHMARK: {b_name.upper()}  (C++ Backbone)")
    print(f"  Variables: {probe_eng.num_vars} | Initial Bias Candidates: {probe_eng.get_bias_count()}")
    print("=" * 80)
    
    algos_to_run = ["pquacq", "quacq", "growacq", "bruteca"] if args.algo == "all" else [args.algo]
    
    results = []
    for algo in algos_to_run:
        print(f"Running {algo.upper()} (max queries = {args.max_queries})...", end="", flush=True)
        res = run_single_algo(b_name, algo, constructor_fn, args.max_queries, args.findc2)
        results.append(res)
        print(f" Done in {res['wall_ms']:.2f} ms! (Learned: {res['active']} active + {res['implied']} implied = {res['total']} total | Queries: {res['total_queries']})", flush=True)
        
    print("\n" + "=" * 80)
    print(f"{'Algorithm':<10} | {'Active':<7} | {'Implied':<8} | {'Total':<6} | {'SAT?':<5} | {'Queries':<8} | {'C++ Time':<10} | {'Wall Time':<10}")
    print("-" * 80)
    for r in results:
        sat_str = "YES" if r["is_sat"] else "NO"
        print(f"{r['algo']:<10} | {r['active']:<7d} | {r['implied']:<8d} | {r['total']:<6d} | {sat_str:<5} | {r['total_queries']:<8d} | {r['cpp_ms']:6.2f} ms | {r['wall_ms']:6.2f} ms")
    print("=" * 80)
    
    if results and results[0]["sol"]:
        sol = results[0]["sol"]
        print(f"\nSample Verified Solution for {b_name.upper()}:")
        if "sudoku" in b_name:
            dim = int(len(sol)**0.5)
            for r in range(dim):
                print(" ", [sol[r * dim + c] for c in range(dim)])
        elif b_name == "zebra":
            cats = ["Nationalities", "Colors", "Cigarettes", "Pets", "Drinks"]
            labels = [
                ["ukr", "norge", "eng", "spain", "jap"],
                ["red", "blue", "yellow", "green", "ivory"],
                ["oldGold", "parly", "kools", "lucky", "chest"],
                ["zebra", "dog", "horse", "fox", "snails"],
                ["coffee", "tea", "h2o", "milk", "oj"]
            ]
            for c_idx in range(5):
                row_vals = {labels[c_idx][i]: sol[c_idx * 5 + i] for i in range(5)}
                print(f"  {cats[c_idx]:<14}: {row_vals}")
        elif "golomb" in b_name:
            print("  Marks on ruler:", sol)
            dists = sorted(abs(sol[j] - sol[i]) for i in range(len(sol)) for j in range(i+1, len(sol)))
            print("  Pairwise distances (all distinct):", dists)
        else:
            print(" ", sol)

if __name__ == "__main__":
    main()
