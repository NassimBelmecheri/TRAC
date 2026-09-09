"""
Shared utilities for the TRAC platform.

This module bootstraps the Python path so that the ``pycona`` package (the
symbolic Constraint-Acquisition engine) is importable, and exposes common
helpers (paths, device selection, seeding).

``pycona`` is discovered automatically whether it is bundled inside the repo
(next to ``trac/``) or lives one directory above the platform.
"""
import os
import sys
import glob
import random
import warnings

import numpy as np

# Silence a noisy (harmless) PyTorch prototype warning triggered by the
# TransformerEncoder padding mask.
warnings.filterwarnings("ignore", message=".*nested tensors.*")


def _register_xpu_runtime():
    """Make the Intel oneAPI/SYCL runtime DLLs discoverable before torch loads.

    Intel Arc (XPU) builds of torch need the SYCL/DPC++ runtime (``sycl8.dll``
    etc.) on the DLL search path. On systems where that runtime ships inside a
    conda environment (e.g. an ``intel_extension_for_pytorch`` env) rather than
    system-wide, torch fails to import with ``WinError 126``. We look for the
    runtime and register it. The directory can be overridden with the
    ``TRAC_XPU_RUNTIME`` environment variable.
    """
    if os.name != "nt":
        return
    candidates = []
    env = os.environ.get("TRAC_XPU_RUNTIME")
    if env:
        candidates.append(env)
    # Common conda-env locations that bundle the Intel GPU runtime.
    home = os.path.expanduser("~")
    candidates += glob.glob(os.path.join(
        home, "AppData", "Local", "anaconda3", "envs", "*", "Library", "bin"))
    candidates += glob.glob(os.path.join(
        home, "AppData", "Local", "miniconda3", "envs", "*", "Library", "bin"))
    for d in candidates:
        if d and os.path.isfile(os.path.join(d, "sycl8.dll")):
            try:
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            except Exception:
                pass
            return d
    return None


_register_xpu_runtime()

# --- Path bootstrap ---------------------------------------------------------
# .../<repo>/trac/utils.py
HERE = os.path.dirname(os.path.abspath(__file__))
PLATFORM_ROOT = os.path.dirname(HERE)             # dir that contains trac/ (repo root)
REPO_ROOT = os.path.dirname(PLATFORM_ROOT)        # one level above (legacy layout)


def _bootstrap_pycona():
    """Add whichever directory contains the ``pycona`` package to ``sys.path``.

    Supports both layouts: ``pycona`` bundled inside the repo (alongside
    ``trac/``) or placed one directory above the platform.
    """
    for base in (PLATFORM_ROOT, REPO_ROOT):
        if os.path.isfile(os.path.join(base, "pycona", "__init__.py")):
            if base not in sys.path:
                sys.path.insert(0, base)
            return base
    # Fallback: keep the legacy behaviour so a pip-installed pycona still works.
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT


_bootstrap_pycona()

# --- Artifact directories ---------------------------------------------------
ARTIFACTS_DIR = os.path.join(PLATFORM_ROOT, "artifacts")
DATA_DIR = os.path.join(ARTIFACTS_DIR, "data")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
RESULTS_DIR = os.path.join(ARTIFACTS_DIR, "results")

for _d in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)


def get_device(prefer_gpu: bool = True) -> str:
    """Return the best available accelerator: 'cuda' (NVIDIA), 'xpu' (Intel Arc),
    else 'cpu'."""
    try:
        import torch
        if prefer_gpu:
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
    except Exception:
        pass
    return "cpu"


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)
    except Exception:
        pass


def data_path(name: str) -> str:
    return os.path.join(DATA_DIR, name)


def safe_filename(name: str) -> str:
    """Make ``name`` safe to use as a file name on all platforms.

    Constraint names such as ``X+Y>Z`` or ``|X-Y|>|Z-T|`` contain characters
    (``> < | ! : * ? " / \\``) that are illegal in Windows paths; replace any
    non ``[A-Za-z0-9._-]`` character with ``_``.
    """
    import re
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def model_path(name: str) -> str:
    return os.path.join(MODELS_DIR, name)


def results_path(name: str) -> str:
    return os.path.join(RESULTS_DIR, name)
