"""
TRAC platform - command-line interface.

Examples
--------
    python cli.py list
    python cli.py generate --benchmark murder --variant TO3 --samples 3000
    python cli.py train    --dataset artifacts/data/murder_paper_TO3.csv --epochs 100
    python cli.py predict  --model artifacts/models/toracle_murder_TO3.pth --benchmark murder
    python cli.py acquire  --model artifacts/models/toracle_murder_TO3.pth --learner FASTCA
    python cli.py run      --benchmark murder --variant TO3 --epochs 100   # full pipeline
"""
import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from trac import (list_benchmarks, generate_dataset, train_oracle,
                  classify_dataset, acquire, VARIANTS, LEARNERS, utils)


def _p(msg, frac=None):
    print("  " + msg)


def cmd_list(_):
    print("Available benchmarks:")
    for name, desc, scales in list_benchmarks():
        print(f"  - {name:16s} scales={scales}  {desc}")
    print(f"\nVariants: {list(VARIANTS)}")
    print(f"Learners: {list(LEARNERS)}")


def cmd_generate(a):
    df, meta = generate_dataset(a.benchmark, scale=a.scale, variant=a.variant,
                               n_samples=a.samples, seed=a.seed, progress=_p)
    out = a.out or utils.data_path(f"{a.benchmark}_{meta['scale']}_{a.variant}.csv")
    df.to_csv(out, index=False)
    print(json.dumps({k: meta[k] for k in
                      ("name", "scale", "variant", "n_samples", "n_positive",
                       "n_negative", "n_vars", "d_max")}, indent=2))
    print(f"Saved -> {out}")
    return out


def cmd_train(a):
    df = pd.read_csv(a.dataset)
    base = os.path.basename(a.dataset).replace(".csv", "").split("_")
    res = train_oracle(df, benchmark=a.benchmark or (base[0] if base else None),
                       scale=a.scale or (base[1] if len(base) > 1 else None),
                       variant=a.variant or (base[-1] if base else None),
                       epochs=a.epochs, batch_size=a.batch, lr=a.lr, progress=_p)
    print(json.dumps({"model_path": res["model_path"],
                      "best_metrics": {k: res["best_metrics"][k] for k in
                                       ("accuracy", "precision", "recall", "f1")},
                      "train_time_s": res["train_time_s"]}, indent=2))
    return res["model_path"]


def cmd_predict(a):
    cfg_path = a.model.replace(".pth", ".json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    df, _ = generate_dataset(a.benchmark or cfg.get("benchmark"),
                             scale=a.scale or cfg.get("scale"),
                             variant=a.variant, n_samples=a.samples, seed=a.seed)
    res = classify_dataset(a.model, df)
    print(json.dumps(res["overall"], indent=2))
    if res.get("groups"):
        print("per-arity:")
        print(json.dumps(res["groups"], indent=2))


def cmd_acquire(a):
    cfg_path = a.model.replace(".pth", ".json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    out = acquire(a.benchmark or cfg.get("benchmark"),
                  scale=a.scale or cfg.get("scale"),
                  model_path=a.model, learner=a.learner,
                  time_limit=a.time_limit, verbose=a.verbose, progress=_p)
    out.pop("learned_constraints", None)
    print(json.dumps(out, indent=2))


def cmd_run(a):
    print("== generate ==")
    ds = cmd_generate(a)
    print("== train ==")
    a.dataset = ds
    model = cmd_train(a)
    print("== acquire ==")
    a.model = model
    cmd_acquire(a)


def cmd_reproduce(a):
    from trac import reproduce as R
    bl = [b.strip() for b in a.benchmarks.split(",")] if a.benchmarks else None
    if a.rq == 1:
        df = R.rq1(benchmarks=bl, time_limit=a.time_limit, progress=_p)
    elif a.rq == 2:
        df = R.rq2(benchmarks=bl, source=a.source, epochs=a.epochs,
                   generate_missing=a.generate_missing, progress=_p)
    elif a.rq == 3:
        df = R.rq3(benchmarks=bl, source=a.source, epochs=a.epochs,
                   time_limit=a.time_limit, progress=_p)
    elif a.rq == 4:
        cons = [c.strip() for c in a.constraints.split(",")] if a.constraints else None
        df = R.rq4(constraints=cons, epochs=a.epochs, progress=_p)
    else:
        raise SystemExit("--rq must be 1, 2, 3 or 4")
    print(f"\n=== RQ{a.rq} results ===")
    print(df.to_string(index=False))
    out = utils.results_path(f"rq{a.rq}_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


def build_parser():
    p = argparse.ArgumentParser(description="TRAC neuro-symbolic CA platform")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)

    def add_common(sp):
        sp.add_argument("--benchmark")
        sp.add_argument("--scale", default=None)
        sp.add_argument("--variant", default="TO3", choices=list(VARIANTS))
        sp.add_argument("--samples", type=int, default=3000)
        sp.add_argument("--seed", type=int, default=42)

    g = sub.add_parser("generate"); add_common(g); g.add_argument("--out")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train")
    t.add_argument("--dataset", required=True)
    t.add_argument("--benchmark"); t.add_argument("--scale"); t.add_argument("--variant")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict"); add_common(pr)
    pr.add_argument("--model", required=True)
    pr.set_defaults(func=cmd_predict)

    ac = sub.add_parser("acquire")
    ac.add_argument("--model", required=True)
    ac.add_argument("--benchmark"); ac.add_argument("--scale")
    ac.add_argument("--learner", default="FASTCA", choices=list(LEARNERS))
    ac.add_argument("--time_limit", type=int, default=10)
    ac.add_argument("--verbose", type=int, default=0)
    ac.set_defaults(func=cmd_acquire)

    r = sub.add_parser("run"); add_common(r)
    r.add_argument("--epochs", type=int, default=100)
    r.add_argument("--batch", type=int, default=128)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--learner", default="FASTCA", choices=list(LEARNERS))
    r.add_argument("--time_limit", type=int, default=10)
    r.add_argument("--verbose", type=int, default=0)
    r.add_argument("--out")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("reproduce", help="Reproduce paper experiments RQ1-RQ4")
    rp.add_argument("--rq", type=int, required=True, choices=[1, 2, 3, 4])
    rp.add_argument("--benchmarks", help="comma-separated benchmark names (default: subset)")
    rp.add_argument("--constraints", help="RQ4 only: comma-separated constraint names")
    rp.add_argument("--source", default="pretrained", choices=["pretrained", "train"],
                    help="RQ2/RQ3: use existing checkpoints or retrain with the "
                         "corrected data generator")
    rp.add_argument("--generate_missing", action="store_true",
                    help="RQ2 only: synthesise data for benchmarks lacking a shipped dataset")
    rp.add_argument("--epochs", type=int, default=60)
    rp.add_argument("--time_limit", type=int, default=8)
    rp.set_defaults(func=cmd_reproduce)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
