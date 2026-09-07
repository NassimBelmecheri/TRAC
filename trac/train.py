"""
Unified trainer for the T-ORACLE (the "train the Toracle" stage).

Replaces the three near-identical ``train_model.py`` scripts (TO1/TO2/TO3).
Trains :class:`CSPAttentionModel` with a binary cross-entropy objective plus an
L1 sparsity penalty on the learned dependency matrix M, sizing the network from
the data and persisting a self-describing checkpoint.
"""
import os
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split

from . import utils
from .model import build_model
from .checkpoint import save_checkpoint


def _loaders(X, y, batch_size, val_split, seed):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y if len(set(y)) > 1 else None
    )
    tr = torch.utils.data.TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    va = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = torch.utils.data.DataLoader(va, batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader


def train_oracle(df, benchmark=None, scale=None, variant=None, model_name=None,
                 epochs=50, batch_size=128, lr=1e-3, l1_lambda=1e-4,
                 embed_dim=32, num_heads=4, val_split=0.2, seed=42,
                 device=None, progress=None):
    """Train a T-ORACLE on the dataset ``df`` (columns ``var_*`` + ``label``).

    :return: a dict with metrics, training history and the checkpoint path.
    """
    utils.set_seed(seed)
    device = device or utils.get_device()

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.int64)

    num_positions = X.shape[1]
    num_values = max(1, int(X.max()))

    tr_loader, va_loader = _loaders(X, y, batch_size, val_split, seed)

    model = build_model(num_values, num_positions, embed_dim, num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    history = []
    best_score = -1.0
    best_metrics = None
    if model_name is None:
        model_name = f"toracle_{benchmark}_{variant}"
    pth = utils.model_path(f"{utils.safe_filename(model_name)}.pth")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            preds, adj_logits, _ = model(bx)
            cls_loss = criterion(preds.squeeze(-1), by.float())
            structure_loss = torch.norm(torch.sigmoid(adj_logits), p=1)
            loss = cls_loss + l1_lambda * structure_loss
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        avg_loss = run_loss / max(1, len(tr_loader))

        val_metrics = _evaluate(model, va_loader, device)
        val_metrics["epoch"] = epoch
        val_metrics["train_loss"] = avg_loss
        history.append(val_metrics)

        score = (val_metrics["recall"] + val_metrics["accuracy"]) / 2.0
        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            save_checkpoint(model, pth, {
                "num_values": num_values,
                "num_positions": num_positions,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "benchmark": benchmark,
                "scale": scale,
                "variant": variant,
                "epoch": epoch,
                "metrics": {k: val_metrics[k] for k in
                            ("accuracy", "precision", "recall", "f1")},
            })

        if progress:
            progress(f"epoch {epoch}/{epochs} | loss {avg_loss:.4f} | "
                     f"val acc {val_metrics['accuracy']:.3f} rec {val_metrics['recall']:.3f}",
                     epoch / epochs)

    elapsed = time.time() - t0
    return {
        "model_path": pth,
        "model_name": model_name,
        "device": device,
        "epochs": epochs,
        "train_time_s": round(elapsed, 2),
        "best_metrics": best_metrics,
        "final_metrics": history[-1] if history else None,
        "history": history,
        "num_values": num_values,
        "num_positions": num_positions,
    }


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for bx, by in loader:
        bx = bx.to(device)
        mask = (bx == 0)
        preds, _, _ = model(bx, None, src_key_padding_mask=mask)
        p = preds.squeeze(-1).cpu().numpy()
        if p.ndim == 0:
            p = np.array([p])
        probs.extend(p.tolist())
        labels.extend(by.numpy().tolist())

    probs = np.array(probs)
    labels = np.array(labels)
    pred = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "confusion": cm.tolist(),
    }
