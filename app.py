"""
TRAC platform - Streamlit UI.

A single web app that walks through the whole neuro-symbolic pipeline:

    1. Generate data   (TO1 / TO2 / TO3 variants)
    2. Train T-ORACLE  (the Transformer neural oracle)
    3. Predict         (classify assignments)
    4. Acquire         (FASTCA + neural oracle -> constraint network)

Run with:  streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import glob
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from trac import (
    list_benchmarks, generate_dataset, train_oracle,
    classify_dataset, classify_assignment, acquire,
    VARIANTS, LEARNERS, utils, reproduce as R,
)
from trac.benchmarks import BENCHMARKS, default_scale, PARAMETRIC
from trac.checkpoint import load_config
from trac.reproduce import LEGACY_TOKENS, CHECKER_CONSTRAINTS, PAPER_BENCHMARKS
from trac import viz

st.set_page_config(page_title="TRAC Platform", page_icon="🧩", layout="wide")

VARIANT_HELP = {
    "TO1": "Full assignments only (balanced solutions / non-solutions).",
    "TO2": "Mix of full and partial assignments of varying scope sizes.",
    "TO3": "Constraint-specific tuples per target scope (best in the paper).",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def scales_for(bench):
    return list(BENCHMARKS[bench]["scales"].keys())


def list_models():
    models = []
    for pth in sorted(glob.glob(utils.model_path("*.pth"))):
        try:
            cfg = load_config(pth)
        except Exception:
            cfg = {}
        models.append((os.path.basename(pth), pth, cfg))
    return models


def confusion_df(cm):
    return pd.DataFrame(cm, index=["true 0", "true 1"], columns=["pred 0", "pred 1"])


def metric_row(m):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.3f}")
    c2.metric("Precision", f"{m['precision']:.3f}")
    c3.metric("Recall", f"{m['recall']:.3f}")
    c4.metric("F1", f"{m['f1']:.3f}")


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🧩 TRAC")
st.sidebar.caption("Transformer-oracle + FASTCA\nNeuro-symbolic constraint acquisition")
st.sidebar.write(f"**Device:** `{utils.get_device()}`")
st.sidebar.write(f"**Models saved:** {len(list_models())}")
st.sidebar.divider()
st.sidebar.markdown(
    "**Pipeline**\n\n"
    "1. Generate data\n2. Train T-ORACLE\n3. Predict\n4. Acquire (FASTCA)"
)

st.title("TRAC — Neuro-Symbolic Constraint Acquisition")

tab_over, tab_gen, tab_train, tab_pred, tab_acq, tab_repro = st.tabs(
    ["📖 Overview", "① Generate", "② Train", "③ Predict", "④ Acquire", "⑤ Reproduce"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with tab_over:
    st.markdown(
        """
This platform unifies the paper *"Learning Symbolic Constraint Representations
from Examples"* into one end-to-end tool.

- **T-ORACLE** — a Transformer that learns to classify (partial) assignments as
  *valid* / *invalid*, emulating a human oracle. Variants **TO1/TO2/TO3** differ
  only in training data.
- **FASTCA** — the time-efficient symbolic learner (paper Algorithm 1) that
  queries the oracle and returns a constraint network.

Data is generated directly from each benchmark's ground-truth oracle, so the
whole loop — **generate → train → predict/acquire** — runs without any external
files.
        """
    )
    st.subheader("Benchmark registry")
    rows = [{"benchmark": n, "scales": ", ".join(s), "description": d}
            for n, d, s in list_benchmarks()]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    with st.expander("Paper ↔ code naming reconciled by this platform"):
        st.markdown(
            "- **FASTCA** (paper) = `BruteCA` (legacy code) → exposed as `trac.FastCA`.\n"
            "- **T-ORACLE** (paper) = `CSPAttentionModel` → unified in `trac/model.py`.\n"
            "- **TO1/TO2/TO3** = three duplicated folders → one `variant` switch in `trac/datagen.py`.\n"
            "- Model dimensions are now derived from the instance / stored in the checkpoint "
            "(no more brittle filename parsing)."
        )

# --------------------------------------------------------------------------- #
# 1. Generate
# --------------------------------------------------------------------------- #
with tab_gen:
    st.header("① Generate training data")
    c1, c2, c3 = st.columns(3)
    g_bench = c1.selectbox("Benchmark", list(BENCHMARKS.keys()), key="g_bench")
    g_scale = c2.selectbox("Scale", scales_for(g_bench), key="g_scale")
    g_variant = c3.selectbox("Variant", list(VARIANTS), key="g_variant",
                             help="\n".join(f"{k}: {v}" for k, v in VARIANT_HELP.items()))
    st.caption(VARIANT_HELP[g_variant])

    c4, c5 = st.columns(2)
    g_samples = c4.slider("Approx. #samples", 500, 12000, 3000, step=500)
    g_seed = c5.number_input("Seed", value=42, step=1)

    # --- Parametric benchmarks (e.g. custom Sudoku dimensions) ---------------
    g_params = None
    if g_bench in PARAMETRIC:
        spec = PARAMETRIC[g_bench]
        with st.expander(f"⚙️ Custom {g_bench} parameters", expanded=(g_bench == "sudoku")):
            use_custom = st.checkbox("Use custom parameters (overrides Scale)",
                                     value=False, key="g_use_custom")
            pcols = st.columns(len(spec["params"]))
            vals = {}
            for (arg, (label, dflt, lo, hi)), col in zip(spec["params"].items(), pcols):
                vals[arg] = col.number_input(label, min_value=lo, max_value=hi,
                                             value=dflt, step=1, key=f"g_p_{arg}")
            if spec.get("note"):
                st.caption(spec["note"])
            if g_bench == "sudoku":
                ok = vals["block_size_row"] * vals["block_size_col"] == vals["grid_size"]
                if use_custom and not ok:
                    st.warning("block rows × block cols must equal grid size for a valid "
                               "Sudoku (e.g. 3×3→9).")
                st.caption(f"→ {vals['grid_size']}×{vals['grid_size']} grid = "
                           f"{vals['grid_size']**2} variables, domain 1..{vals['grid_size']}")
            if use_custom:
                g_params = vals

    if st.button("Generate data", type="primary"):
        status = st.status("Generating ...", expanded=True)
        log = status.empty()

        def prog(msg, frac=None):
            log.write(msg)

        try:
            df, meta = generate_dataset(g_bench, scale=g_scale, variant=g_variant,
                                        n_samples=int(g_samples), seed=int(g_seed),
                                        params=g_params, progress=prog)
            tag = meta["scale"]
            out = utils.data_path(f"{g_bench}_{tag}_{g_variant}.csv")
            df.to_csv(out, index=False)
            st.session_state["dataset"] = df
            st.session_state["dataset_meta"] = meta
            st.session_state["dataset_path"] = out
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Failed", state="error")
            st.exception(e)

    if "dataset_meta" in st.session_state:
        meta = st.session_state["dataset_meta"]
        st.success(f"Dataset ready: **{meta['n_samples']}** rows "
                   f"({meta['n_positive']} valid / {meta['n_negative']} invalid), "
                   f"{meta['n_vars']} variables, domain 1..{meta['d_max']}.")
        st.caption(f"Saved to `{st.session_state['dataset_path']}`")
        cc1, cc2 = st.columns([2, 1])
        cc1.dataframe(st.session_state["dataset"].head(12), width='stretch')
        dist = st.session_state["dataset"]["label"].value_counts().rename(
            {0: "invalid", 1: "valid"})
        cc2.bar_chart(dist)

# --------------------------------------------------------------------------- #
# 2. Train
# --------------------------------------------------------------------------- #
with tab_train:
    st.header("② Train the T-ORACLE")

    source = st.radio("Training data source", ["From last generated", "From saved CSV"],
                      horizontal=True)
    df = None
    meta = st.session_state.get("dataset_meta", {})
    if source == "From last generated":
        if "dataset" in st.session_state:
            df = st.session_state["dataset"]
            st.caption(f"Using generated **{meta.get('name')}** "
                       f"[{meta.get('scale')}] / {meta.get('variant')} "
                       f"({meta.get('n_samples')} rows)")
        else:
            st.info("Generate a dataset first (tab ①).")
    else:
        csvs = sorted(glob.glob(utils.data_path("*.csv")))
        if csvs:
            pick = st.selectbox("Saved dataset", csvs,
                                format_func=os.path.basename)
            df = pd.read_csv(pick)
            base = os.path.basename(pick).replace(".csv", "").split("_")
            meta = {"name": base[0] if base else "custom",
                    "scale": base[1] if len(base) > 1 else None,
                    "variant": base[-1] if base else None}
            st.caption(f"{len(df)} rows, {df.shape[1]-1} variables")
        else:
            st.info("No saved datasets found.")

    c1, c2, c3 = st.columns(3)
    epochs = c1.slider("Epochs", 5, 500, 60, step=5)
    batch = c2.select_slider("Batch size", [32, 64, 128, 256], value=128)
    lr = c3.select_slider("Learning rate", [1e-4, 5e-4, 1e-3, 2e-3], value=1e-3)

    if st.button("Train", type="primary", disabled=df is None):
        prog_bar = st.progress(0.0, text="starting ...")

        def prog(msg, frac=None):
            prog_bar.progress(min(1.0, frac or 0.0), text=msg)

        try:
            res = train_oracle(df, benchmark=meta.get("name"), scale=meta.get("scale"),
                               variant=meta.get("variant"), epochs=int(epochs),
                               batch_size=int(batch), lr=float(lr), progress=prog)
            prog_bar.progress(1.0, text="done")
            st.session_state["train_res"] = res
        except Exception as e:
            st.exception(e)

    if "train_res" in st.session_state:
        res = st.session_state["train_res"]
        bm = res["best_metrics"]
        st.success(f"Best checkpoint saved to `{res['model_path']}` "
                   f"({res['train_time_s']}s on {res['device']})")

        # --- headline metrics ---
        metric_row(bm)

        # --- collapse warning (all-one-class) ---
        cm = bm["confusion"]
        if bm["recall"] == 0.0 or bm["recall"] == 1.0 or abs(bm["accuracy"] - 0.5) < 1e-6:
            st.warning("⚠️ The model looks **collapsed to a single class** "
                       "(recall 0 or 1, accuracy ≈ 0.5). This usually means it hasn't "
                       "converged yet — train more epochs, use a smaller **scale**, or "
                       "the **TO3** variant. Large paper-scale problems need many epochs.")

        st.divider()
        hist = res["history"]
        labels = res.get("val_labels")
        probs = res.get("val_probs")

        # --- curves row: metrics + loss ---
        c1, c2 = st.columns(2)
        c1.pyplot(viz.metric_curves_fig(hist))
        c2.pyplot(viz.loss_curve_fig(hist))

        # --- confusion matrices (counts + normalised) ---
        st.subheader("Confusion matrix")
        c3, c4 = st.columns(2)
        c3.pyplot(viz.confusion_heatmap(cm, normalize=False))
        c4.pyplot(viz.confusion_heatmap(cm, normalize=True))

        # --- ROC / PR / score distribution (need probabilities) ---
        if labels is not None and probs is not None:
            st.subheader("Discrimination")
            c5, c6, c7 = st.columns(3)
            c5.pyplot(viz.roc_curve_fig(labels, probs))
            c6.pyplot(viz.pr_curve_fig(labels, probs))
            c7.pyplot(viz.prob_hist_fig(labels, probs))

        # --- derived statistics table ---
        st.subheader("Detailed statistics")
        stats = viz.derived_stats(cm, labels, probs)
        sc1, sc2 = st.columns(2)
        with sc1:
            st.dataframe(pd.DataFrame(
                [(k, stats[k]) for k in ("TP", "TN", "FP", "FN")],
                columns=["count", "value"]), width='stretch', hide_index=True)
        with sc2:
            st.dataframe(pd.DataFrame(
                [(k, v) for k, v in stats.items() if k not in ("TP", "TN", "FP", "FN")],
                columns=["metric", "value"]), width='stretch', hide_index=True)

        with st.expander("Per-epoch history (raw)"):
            st.dataframe(pd.DataFrame(hist)[
                ["epoch", "train_loss", "accuracy", "precision", "recall", "f1"]],
                width='stretch', hide_index=True)

# --------------------------------------------------------------------------- #
# 3. Predict
# --------------------------------------------------------------------------- #
with tab_pred:
    st.header("③ Predict — classify assignments")
    models = list_models()
    if not models:
        st.info("Train a model first (tab ②).")
    else:
        names = [m[0] for m in models]
        pick = st.selectbox("Model", names, key="pred_model")
        model_path = dict((m[0], m[1]) for m in models)[pick]
        cfg = dict((m[0], m[2]) for m in models)[pick]
        st.caption(f"benchmark=`{cfg.get('benchmark')}` scale=`{cfg.get('scale')}` "
                   f"variant=`{cfg.get('variant')}` · {cfg.get('num_positions')} vars, "
                   f"domain 1..{cfg.get('num_values')}")

        mode = st.radio("Mode", ["Evaluate a generated test set", "Single assignment"],
                        horizontal=True)

        if mode == "Evaluate a generated test set":
            c1, c2, c3 = st.columns(3)
            t_variant = c1.selectbox("Test variant", list(VARIANTS), index=2)
            t_samples = c2.slider("#test samples", 500, 6000, 2000, step=500)
            t_seed = c3.number_input("Test seed", value=7, step=1)
            if st.button("Evaluate", type="primary"):
                with st.spinner("Generating test set and scoring ..."):
                    try:
                        tdf, _ = generate_dataset(cfg.get("benchmark"),
                                                  scale=cfg.get("scale"),
                                                  variant=t_variant,
                                                  n_samples=int(t_samples),
                                                  seed=int(t_seed))
                        res = classify_dataset(model_path, tdf)
                        st.subheader("Overall")
                        metric_row(res["overall"])
                        st.dataframe(confusion_df(res["overall"]["confusion"]))
                        if res.get("groups"):
                            st.subheader("Per-arity (scope size)")
                            grp = pd.DataFrame(res["groups"]).T[
                                ["count", "accuracy", "precision", "recall", "f1"]]
                            st.dataframe(grp, width='stretch')
                    except Exception as e:
                        st.exception(e)
        else:
            n = cfg.get("num_positions", 0)
            dmax = cfg.get("num_values", 9)
            st.caption(f"Enter {n} values in 0..{dmax} (0 = unassigned). "
                       "Comma or space separated.")
            default = ",".join(["0"] * n)
            raw = st.text_area("Assignment vector", value=default, height=80)
            if st.button("Classify", type="primary"):
                try:
                    vals = [int(x) for x in raw.replace(",", " ").split()]
                    out = classify_assignment(model_path, vals)
                    color = "🟢" if out["prediction"] == 1 else "🔴"
                    st.metric("Prediction", f"{color} {out['label']}",
                              f"p={out['probability']:.3f}")
                except Exception as e:
                    st.exception(e)

# --------------------------------------------------------------------------- #
# 4. Acquire
# --------------------------------------------------------------------------- #
with tab_acq:
    st.header("④ Acquire — FASTCA + neural oracle")
    models = list_models()
    if not models:
        st.info("Train a model first (tab ②).")
    else:
        names = [m[0] for m in models]
        pick = st.selectbox("Trained oracle", names, key="acq_model")
        model_path = dict((m[0], m[1]) for m in models)[pick]
        cfg = dict((m[0], m[2]) for m in models)[pick]
        bench = cfg.get("benchmark")
        scale = cfg.get("scale")
        st.caption(f"Will acquire the **{bench}** [{scale}] network "
                   f"({cfg.get('num_positions')} variables).")

        c1, c2 = st.columns(2)
        learner = c1.selectbox("Symbolic learner", list(LEARNERS))
        tlimit = c2.slider("Per-query time limit (s)", 1, 20, 5)

        if bench is None:
            st.warning("This checkpoint has no benchmark metadata; retrain via tab ②.")
        elif st.button("Run acquisition", type="primary"):
            status = st.status("Acquiring ...", expanded=True)
            log = status.empty()
            try:
                out = acquire(bench, scale=scale, model_path=model_path,
                              learner=learner, time_limit=int(tlimit),
                              progress=lambda m, f=None: log.write(m))
                status.update(label="Acquisition complete", state="complete")
                st.session_state["acq_out"] = out
            except Exception as e:
                status.update(label="Failed", state="error")
                st.exception(e)

        if "acq_out" in st.session_state:
            out = st.session_state["acq_out"]
            ev = out["evaluation"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{ev['precision']:.3f}")
            c2.metric("Recall", f"{ev['recall']:.3f}")
            c3.metric("F1", f"{ev['f1']:.3f}")
            c4.metric("Queries", out["membership_queries"])
            c5, c6, c7 = st.columns(3)
            c5.metric("Learned constraints", out["n_learned"])
            c6.metric("Target constraints", out["n_target"])
            c7.metric("Wall time (s)", out["wall_time_s"])
            st.caption(f"bias size: {out['bias_size']} · learner: {out['learner']}")
            with st.expander(f"Learned constraints ({out['n_learned']})"):
                st.code("\n".join(out["learned_constraints"]) or "(none)")

# --------------------------------------------------------------------------- #
# 5. Reproduce
# --------------------------------------------------------------------------- #
with tab_repro:
    st.header("⑤ Reproduce the paper (RQ1–RQ4)")
    st.caption("Regenerates the paper's tables, reusing the datasets and "
               "pretrained checkpoints already in the repository where available.")

    rq = st.radio(
        "Research question", [
            "RQ1 · learner efficiency (Table 1)",
            "RQ2 · oracle classification (Table 2)",
            "RQ3 · learner × oracle acquisition (Table 3)",
            "RQ4 · constraint checker (Table 4)",
        ], index=1)
    all_benches = list(PAPER_BENCHMARKS)
    st.caption("Benchmark set = the paper's 10 (nqueens / golomb are extra "
               "pycona benchmarks, not evaluated in the paper).")

    # ---- RQ1 ---------------------------------------------------------------
    if rq.startswith("RQ1"):
        st.markdown("Runs each learner with the **ground-truth** oracle and reports "
                    "query count, per-query time and total time. FASTCA trades more "
                    "queries for a much lower per-query cost.")
        bl = st.multiselect("Benchmarks", all_benches, default=["murder", "zebra"])
        tl = st.slider("Per-query time limit (s)", 1, 20, 5, key="rq1_tl")
        if st.button("Run RQ1", type="primary"):
            status = st.status("Running RQ1 ...", expanded=True)
            log = status.empty()
            try:
                df = R.rq1(benchmarks=bl, time_limit=int(tl),
                           progress=lambda m: log.write(str(m)))
                status.update(label="Done", state="complete")
                st.dataframe(df, width='stretch', hide_index=True)
            except Exception as e:
                status.update(label="Failed", state="error"); st.exception(e)

    # ---- RQ2 ---------------------------------------------------------------
    elif rq.startswith("RQ2"):
        st.markdown("Classification **Accuracy / Recall / Precision** (`Acc/Rec/Prec`) "
                    "of TO1/TO2/TO3 on each benchmark's held-out data.")
        bl = st.multiselect("Benchmarks", all_benches, default=all_benches)
        src = st.radio("Source", ["pretrained", "train"], horizontal=True,
                       help="pretrained = load existing checkpoints (fast); "
                            "train = retrain from the existing datasets")
        gen_missing = st.checkbox("Synthesise data for benchmarks with no shipped "
                                  "TO3 dataset (slower)", value=False)
        ep = st.slider("Epochs (train mode)", 10, 200, 60, key="rq2_ep")
        if st.button("Run RQ2", type="primary"):
            status = st.status("Running RQ2 ...", expanded=True)
            log = status.empty()
            try:
                df = R.rq2(benchmarks=bl, source=src, epochs=int(ep),
                           generate_missing=gen_missing,
                           progress=lambda m: log.write(str(m)))
                status.update(label="Done", state="complete")
                st.dataframe(df, width='stretch', hide_index=True)
                st.caption("Each cell is Acc/Rec/Prec (%).")
            except Exception as e:
                status.update(label="Failed", state="error"); st.exception(e)

    # ---- RQ3 ---------------------------------------------------------------
    elif rq.startswith("RQ3"):
        st.markdown("Acquired-network **precision / recall / F1** for each "
                    "learner × oracle pairing.")
        bl = st.multiselect("Benchmarks", all_benches, default=["murder", "zebra"])
        variants = st.multiselect("Oracle variants", list(VARIANTS),
                                   default=["TO3"])
        learners = st.multiselect("Learners", list(LEARNERS),
                                   default=["FASTCA"])
        c1, c2 = st.columns(2)
        src3 = c1.radio("Oracle source", ["pretrained", "train"], horizontal=True,
                        help="pretrained = shipped legacy checkpoints (fast, but "
                             "predate the data-gen fix); train = retrain oracles "
                             "with the corrected generator (reproduces the paper)")
        ep3 = c2.slider("Epochs (train mode)", 20, 200, 100, key="rq3_ep")
        tl = st.slider("Per-query time limit (s)", 1, 20, 6, key="rq3_tl")
        if st.button("Run RQ3", type="primary"):
            status = st.status("Running RQ3 ...", expanded=True)
            log = status.empty()
            try:
                df = R.rq3(benchmarks=bl, variants=tuple(variants),
                           learners=tuple(learners), source=src3, epochs=int(ep3),
                           time_limit=int(tl),
                           progress=lambda m: log.write(str(m)))
                status.update(label="Done", state="complete")
                st.dataframe(df, width='stretch', hide_index=True)
            except Exception as e:
                status.update(label="Failed", state="error"); st.exception(e)

    # ---- RQ4 ---------------------------------------------------------------
    else:
        st.markdown("Trains TO3 to emulate a **symbolic constraint checker** on "
                    "individual constraints of increasing arity.")
        names = [c[1] for c in CHECKER_CONSTRAINTS]
        cons = st.multiselect("Constraints", names, default=["X!=Y", "X+Y>Z"])
        ep = st.slider("Epochs", 20, 150, 50, key="rq4_ep")
        if st.button("Run RQ4", type="primary"):
            status = st.status("Running RQ4 ...", expanded=True)
            log = status.empty()
            try:
                df = R.rq4(constraints=cons, epochs=int(ep),
                           progress=lambda m: log.write(str(m)))
                status.update(label="Done", state="complete")
                st.dataframe(df, width='stretch', hide_index=True)
            except Exception as e:
                status.update(label="Failed", state="error"); st.exception(e)
