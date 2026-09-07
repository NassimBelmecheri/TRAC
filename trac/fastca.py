"""
FASTCA - the time-efficient symbolic CA learner (paper Algorithm 1).

In the paper the exhaustive learner is called **FASTCA**; in the code base it is
implemented as ``pycona.active_algorithms.BruteCA``.  This module simply exposes
the paper-faithful name.

The robustness / correctness fixes now live in ``BruteCA`` itself
(``pycona/active_algorithms/bruteca.py``):

  * ``_generate_violation`` is bounded (no more ``while True`` infinite loop on
    implied constraints) and generates counter-examples that **satisfy the
    learned network L** while violating the target constraint - paper Algorithm 1:
    "generate e that satisfies L and violates c".
  * ``learn`` processes the bias in increasing scope-size order, so unary
    sub-scope constraints are learned before binary ones. This removes the
    spurious-constraint problem on benchmarks with unary constraints (e.g.
    Zebra's ``milk == 3`` / ``norway == 1``), where a unary violation was
    previously misattributed to a binary constraint.
"""
from . import utils  # noqa: F401  (bootstraps pycona)
from pycona.active_algorithms import BruteCA


class FastCA(BruteCA):
    """FASTCA learner (paper Algorithm 1). Alias for the fixed ``BruteCA``."""
    pass


# Paper-name alias.
FASTCA = FastCA

