"""
Prediction / classification with a trained T-ORACLE.

Two entry points:
  * :func:`classify_dataset` - score a dataframe of assignments, reporting overall
    and per-arity (scope-size) metrics, mirroring the paper's RQ2/RQ4 analysis.
  * :func:`classify_assignment` - score a single (partial) assignment vector.
"""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from . import utils
from .checkpoint import load_checkpoint


@torch.no_grad()
def _predict_proba(model, X, device, batch_size=256):
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        bx = torch.from_numpy(X[i:i + batch_size]).to(device)
        mask = (bx == 0)
        preds, _, _ = model(bx, None, src_key_padding_mask=mask)
        p = preds.squeeze(-1).cpu().numpy()
        if p.ndim == 0:
            p = np.array([p])
        probs.extend(p.tolist())
    return np.array(probs)


def _metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion": cm.tolist(),
    }


def classify_dataset(model_path, df, device=None, threshold=0.5):
    """Classify a dataframe (``var_*`` columns, optional ``label``).

    :return: dict with ``overall`` metrics, per-arity ``groups`` and raw ``probs``.
    """
    device = device or utils.get_device()
    model, config = load_checkpoint(model_path, device=device)

    has_label = "label" in df.columns
    feat = df.drop(columns=["label"]) if has_label else df
    X = feat.values.astype(np.float32)

    # Guard: pad/truncate to the model's expected number of positions.
    n_expected = config["num_positions"]
    if X.shape[1] != n_expected:
        raise ValueError(
            f"Dataset has {X.shape[1]} variables but the model expects {n_expected}."
        )

    probs = _predict_proba(model, X, device)
    preds = (probs >= threshold).astype(int)

    result = {"config": config, "probs": probs.tolist(), "preds": preds.tolist()}

    if has_label:
        y = df["label"].values.astype(int)
        result["overall"] = _metrics(y, preds)

        arity = np.count_nonzero(X, axis=1)
        groups = {}
        for a in sorted(set(arity.tolist())):
            m = arity == a
            if m.sum() >= 5:
                groups[f"arity_{a}"] = _metrics(y[m], preds[m])
        result["groups"] = groups
    return result


def classify_assignment(model_path, values, device=None, threshold=0.5):
    """Classify a single assignment given as a length-n list (0 = unassigned)."""
    device = device or utils.get_device()
    model, config = load_checkpoint(model_path, device=device)
    x = np.array(values, dtype=np.float32).reshape(1, -1)
    if x.shape[1] != config["num_positions"]:
        raise ValueError(
            f"Assignment has {x.shape[1]} entries but the model expects "
            f"{config['num_positions']}."
        )
    prob = float(_predict_proba(model, x, device)[0])
    return {"probability": prob, "prediction": int(prob >= threshold),
            "label": "valid" if prob >= threshold else "invalid"}
