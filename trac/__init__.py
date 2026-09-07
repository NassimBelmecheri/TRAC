"""
TRAC - Transformer-oracle + FASTCA: a unified neuro-symbolic constraint
acquisition platform.

Pipeline:  generate data  ->  train the T-ORACLE  ->  predict / acquire.

Public API::

    from trac import (
        list_benchmarks, build_benchmark,      # benchmarks
        generate_dataset,                      # generate data
        train_oracle,                          # train the T-ORACLE
        classify_dataset, classify_assignment, # predict (classification)
        acquire, evaluate_network,             # predict (constraint network)
        TransformerOracle, FastCA,
    )
"""
from . import utils  # noqa: F401  (path bootstrap first)

from .benchmarks import list_benchmarks, build_benchmark, BENCHMARKS, default_scale
from .datagen import generate_dataset, VARIANTS
from .train import train_oracle
from .predict import classify_dataset, classify_assignment
from .oracle import TransformerOracle
from .fastca import FastCA, FASTCA
from .acquire import acquire, evaluate_network, LEARNERS
from . import reproduce
from .reproduce import (rq1, rq2, rq3, rq4, LEGACY_TOKENS, CHECKER_CONSTRAINTS,
                        PAPER_BENCHMARKS)

__version__ = "1.0.0"

__all__ = [
    "list_benchmarks", "build_benchmark", "BENCHMARKS", "default_scale",
    "generate_dataset", "VARIANTS",
    "train_oracle",
    "classify_dataset", "classify_assignment",
    "TransformerOracle", "FastCA", "FASTCA",
    "acquire", "evaluate_network", "LEARNERS",
    "reproduce", "rq1", "rq2", "rq3", "rq4", "LEGACY_TOKENS", "CHECKER_CONSTRAINTS",
    "PAPER_BENCHMARKS",
    "utils", "__version__",
]
