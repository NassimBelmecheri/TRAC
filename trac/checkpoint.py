"""
Checkpoint I/O for T-ORACLE models.

A checkpoint is a pair of files that live side by side in ``artifacts/models``:
  * ``<name>.pth``  - the model ``state_dict``
  * ``<name>.json`` - the configuration needed to rebuild the exact architecture
                      (dimensions, benchmark, variant, metrics ...)

Storing the dimensions explicitly fixes a real inconsistency in the legacy code,
where ``test_model_perscope*.py`` / ``TRAC_test.py`` had to *parse the file name*
or *inspect tensor shapes* to guess ``n_vars`` / ``grid_size``.
"""
import json
import os
import time

import torch

from .model import build_model


def save_checkpoint(model, path_pth, config: dict):
    """Save ``model`` weights to ``path_pth`` and ``config`` to the sibling JSON.

    Robust to (a) accelerator tensors - the ``state_dict`` is moved to CPU first,
    and (b) transient Windows file locks (e.g. an AV/indexer scanning the freshly
    written ``.pth`` between the repeated best-model saves during training) - the
    weights are written to a temp file and atomically renamed, with retries.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_pth)), exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp = path_pth + ".tmp"
    last = None
    for _ in range(6):
        try:
            torch.save(state, tmp)
            os.replace(tmp, path_pth)
            last = None
            break
        except (OSError, RuntimeError) as e:
            last = e
            time.sleep(0.5)
    if last is not None:
        raise last
    with open(_json_path(path_pth), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_config(path_pth: str) -> dict:
    """Return the checkpoint config.

    Falls back to inferring the architecture from the tensor shapes when the
    sibling JSON is absent - this lets the platform load the *legacy* pretrained
    ``.pth`` files shipped in ``TO1/``, ``TO2/`` and ``TO3/`` (which have no
    sidecar metadata) transparently.
    """
    jpath = _json_path(path_pth)
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as f:
            return json.load(f)
    return infer_config(path_pth)


def infer_config(path_pth: str) -> dict:
    """Infer ``{num_values, num_positions, embed_dim, num_heads}`` from weights."""
    state = torch.load(path_pth, map_location="cpu")
    cfg = {
        "num_values": int(state["value_embed.weight"].shape[0]) - 1,
        "num_positions": int(state["pos_embed"].shape[1]),
        "embed_dim": int(state["pos_embed"].shape[2]),
        "num_heads": 4,  # not recoverable from the state_dict; the legacy default
        "legacy": True,
    }
    return cfg


def load_checkpoint(path_pth: str, device: str = "cpu"):
    """Rebuild a model from its checkpoint and load the weights.

    :return: ``(model, config)``
    """
    config = load_config(path_pth)
    model = build_model(
        num_values=config["num_values"],
        num_positions=config["num_positions"],
        embed_dim=config.get("embed_dim", 32),
        num_heads=config.get("num_heads", 4),
    )
    state = torch.load(path_pth, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config


def _json_path(path_pth: str) -> str:
    base, _ = os.path.splitext(path_pth)
    return base + ".json"
