"""
Matplotlib visualisations for the TRAC UI (training diagnostics).

Each function returns a ``matplotlib.figure.Figure`` that Streamlit renders with
``st.pyplot``. Kept dependency-light (matplotlib + scikit-learn only).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, matthews_corrcoef)


_LABELS = ["invalid (0)", "valid (1)"]


def confusion_heatmap(cm, normalize=False):
    """Annotated confusion-matrix heatmap."""
    cm = np.array(cm, dtype=float)
    disp = cm.copy()
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        disp = np.divide(cm, row, out=np.zeros_like(cm), where=row != 0)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() if disp.max() else 1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(_LABELS); ax.set_yticklabels(_LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix" + (" (normalised)" if normalize else ""))
    thresh = disp.max() / 2 if disp.max() else 0.5
    for i in range(2):
        for j in range(2):
            txt = f"{disp[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if disp[i, j] > thresh else "black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def roc_curve_fig(labels, probs):
    labels = np.array(labels); probs = np.array(probs)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    if len(np.unique(labels)) < 2:
        ax.text(0.5, 0.5, "ROC needs both classes", ha="center", va="center")
        ax.set_axis_off(); return fig
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def pr_curve_fig(labels, probs):
    labels = np.array(labels); probs = np.array(probs)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    if len(np.unique(labels)) < 2:
        ax.text(0.5, 0.5, "PR needs both classes", ha="center", va="center")
        ax.set_axis_off(); return fig
    prec, rec, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)
    ax.plot(rec, prec, color="#d62728", lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve"); ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def metric_curves_fig(history):
    """Accuracy / precision / recall / F1 vs epoch."""
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key, color in [("accuracy", "#1f77b4"), ("precision", "#2ca02c"),
                       ("recall", "#ff7f0e"), ("f1", "#9467bd")]:
        ax.plot(ep, [h[key] for h in history], label=key, color=color, lw=1.8)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Validation metrics"); ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def loss_curve_fig(history):
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(ep, [h["train_loss"] for h in history], color="#d62728", lw=1.8)
    ax.axhline(np.log(2), ls="--", color="grey", lw=1, label="random (ln 2)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Train loss")
    ax.set_title("Training loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def prob_hist_fig(labels, probs, threshold=0.5):
    labels = np.array(labels); probs = np.array(probs)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bins = np.linspace(0, 1, 31)
    ax.hist(probs[labels == 0], bins=bins, alpha=0.6, label="invalid (0)", color="#1f77b4")
    ax.hist(probs[labels == 1], bins=bins, alpha=0.6, label="valid (1)", color="#ff7f0e")
    ax.axvline(threshold, ls="--", color="black", lw=1, label=f"threshold={threshold}")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Count")
    ax.set_title("Score distribution by class"); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def derived_stats(cm, labels=None, probs=None):
    """Return a dict of derived classification statistics from a 2×2 CM."""
    cm = np.array(cm)
    tn, fp, fn, tp = cm.ravel()
    total = max(1, tn + fp + fn + tp)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0           # recall / sensitivity
    tnr = tn / (tn + fp) if (tn + fp) else 0.0           # specificity
    ppv = tp / (tp + fp) if (tp + fp) else 0.0           # precision
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = 2 * ppv * tpr / (ppv + tpr) if (ppv + tpr) else 0.0
    acc = (tp + tn) / total
    bal_acc = (tpr + tnr) / 2
    stats = {
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "Accuracy": round(acc, 4),
        "Balanced acc.": round(bal_acc, 4),
        "Precision (PPV)": round(ppv, 4),
        "Recall (TPR)": round(tpr, 4),
        "Specificity (TNR)": round(tnr, 4),
        "NPV": round(npv, 4),
        "F1": round(f1, 4),
    }
    if labels is not None and probs is not None:
        lab = np.array(labels)
        if len(np.unique(lab)) == 2:
            pred = (np.array(probs) >= 0.5).astype(int)
            try:
                stats["MCC"] = round(float(matthews_corrcoef(lab, pred)), 4)
                stats["ROC AUC"] = round(float(auc(*roc_curve(lab, probs)[:2])), 4)
                stats["Avg. precision"] = round(float(average_precision_score(lab, probs)), 4)
            except Exception:
                pass
    return stats
