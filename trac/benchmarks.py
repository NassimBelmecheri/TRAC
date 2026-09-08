"""
Benchmark registry for the TRAC platform.

Thin, unified wrapper around ``pycona.benchmarks`` that:
  * exposes a single ``BENCHMARKS`` registry (name -> constructor + scale presets),
  * derives the model dimensions (n variables, max domain value) *from the
    constructed instance* instead of hard-coding them (the legacy
    ``TRAC_test.py`` used brittle hard-coded ``MODEL_CONFIGS``), and
  * returns a ground-truth :class:`ConstraintOracle` used both to generate data
    and to evaluate the acquired network.

Each benchmark offers a fast ``demo`` scale (interactive use) and, where
meaningful, a larger ``paper`` scale matching the paper's instances.
"""
from functools import partial

from . import utils  # noqa: F401  (ensures pycona is importable)

from pycona.benchmarks import (
    construct_sudoku,
    construct_jsudoku,
    construct_latin_squares,
    construct_nqueens_problem,
    construct_job_shop_scheduling_problem,
    construct_murder_problem,
    construct_examtt_simple,
    construct_nurse_rostering,
    construct_zebra_problem,
    construct_golomb,
    construct_random122,
    construct_random495,
)
from pycona.utils import get_scope


# name -> {desc, scales: {scale_name: constructor(callable -> (instance, oracle))}}
BENCHMARKS = {
    "sudoku": {
        "desc": "Sudoku (AllDifferent rows/cols/blocks).",
        "scales": {
            "demo": partial(construct_sudoku, 2, 2, 4),     # 16 vars, domain 1..4
            "paper": partial(construct_sudoku, 3, 3, 9),    # 81 vars, domain 1..9
        },
    },
    "jsudoku": {
        "desc": "Jigsaw Sudoku (9x9 with irregular blocks).",
        "scales": {"paper": construct_jsudoku},             # 81 vars
    },
    "latin_squares": {
        "desc": "Latin square (each symbol once per row/col).",
        "scales": {
            "demo": partial(construct_latin_squares, 4),    # 16 vars
            "paper": partial(construct_latin_squares, 10),  # 100 vars
        },
    },
    "nqueens": {
        "desc": "N-Queens (AllDifferent + diagonal constraints).",
        "scales": {
            "demo": partial(construct_nqueens_problem, 6),
            "paper": partial(construct_nqueens_problem, 8),
        },
    },
    "job_shop": {
        "desc": "Job-shop scheduling (precedence / no-overlap).",
        "scales": {
            "demo": partial(construct_job_shop_scheduling_problem, 3, 3, 10, 0),
            "paper": partial(construct_job_shop_scheduling_problem, 10, 3, 20, 0),
        },
    },
    "murder": {
        "desc": "Murder logic puzzle (unique solution, 20 vars).",
        "scales": {"paper": construct_murder_problem},
    },
    "exam_timetabling": {
        "desc": "Exam timetabling (different-slot / same-day constraints).",
        "scales": {
            "demo": partial(construct_examtt_simple, 4, 3, 3, 6),
            "paper": partial(construct_examtt_simple, 6, 3, 3, 10),
        },
    },
    "nurse_rostering": {
        "desc": "Nurse rostering (shifts x days x nurses).",
        "scales": {
            "demo": partial(construct_nurse_rostering, 3, 3, 8, 3),
            "paper": partial(construct_nurse_rostering, 3, 7, 18, 5),
        },
    },
    "zebra": {
        "desc": "Zebra puzzle (unique solution, 25 vars).",
        "scales": {"paper": construct_zebra_problem},
    },
    "golomb": {
        "desc": "Golomb ruler (strictly increasing marks).",
        "scales": {"paper": partial(construct_golomb, 8)},
    },
    "random122": {
        "desc": "Random binary network (122 constraints).",
        "scales": {"paper": construct_random122},
    },
    "random495": {
        "desc": "Random binary network (495 constraints).",
        "scales": {"paper": construct_random495},
    },
}


def list_benchmarks():
    """Return ``[(name, desc, [scales...]), ...]``."""
    return [(k, v["desc"], list(v["scales"].keys())) for k, v in BENCHMARKS.items()]


def default_scale(name: str) -> str:
    scales = BENCHMARKS[name]["scales"]
    return "demo" if "demo" in scales else next(iter(scales))


# Benchmarks whose instance can be customised with user-supplied parameters.
# name -> {"fn": constructor, "params": {arg: (label, default, min, max)}}
PARAMETRIC = {
    "sudoku": {
        "fn": construct_sudoku,
        "params": {
            "block_size_row": ("Block rows", 2, 1, 4),
            "block_size_col": ("Block cols", 2, 1, 4),
            "grid_size":      ("Grid size (n×n)", 4, 4, 16),
        },
        "note": "For a valid Sudoku, block rows × block cols must equal the grid size "
                "(e.g. 2×2→4, 2×3→6, 3×3→9, 3×4→12, 4×4→16).",
    },
}


def build_benchmark(name: str, scale: str = None, params: dict = None):
    """Construct a benchmark instance.

    :param params: optional constructor overrides for a parametric benchmark
        (currently ``sudoku``); when given, ``scale`` is ignored.
    :return: ``(instance, oracle, meta)`` where ``meta`` holds derived
        dimensions used to size the model.
    """
    if name not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{name}'. Available: {list(BENCHMARKS)}")

    if params and name in PARAMETRIC:
        instance, oracle = PARAMETRIC[name]["fn"](**params)
        scale = "custom"
    else:
        scales = BENCHMARKS[name]["scales"]
        if scale is None or scale not in scales:
            scale = default_scale(name) if scale is None else next(iter(scales))
        instance, oracle = scales[scale]()

    X = list(instance.X)
    n_vars = len(X)
    d_max = max(int(v.get_bounds()[1]) for v in X)
    d_min = min(int(v.get_bounds()[0]) for v in X)

    # Number of distinct scopes covered by the target constraints.
    target_scopes = {tuple(sorted(v.name for v in get_scope(c))) for c in oracle.constraints}

    meta = {
        "name": name,
        "scale": scale,
        "n_vars": n_vars,
        "d_min": d_min,
        "d_max": d_max,
        "num_target_constraints": len(oracle.constraints),
        "num_target_scopes": len(target_scopes),
        "params": params,
    }
    return instance, oracle, meta
