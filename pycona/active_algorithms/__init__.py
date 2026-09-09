"""
This module imports various active_algorithms for ICA implementations:
 - QuAcq:
 - MQuAcq:
 - MQuAcq2:
 - GrowAcq:
"""

from .algorithm_core import AlgorithmCAInteractive
from .quacq import QuAcq
from .mquacq2 import MQuAcq2
from .mquacq import MQuAcq
from .growacq import GrowAcq
from .pquacq import PQuAcq
from .mineacq import MineAcq
from .genacq import GenAcq
from .bruteca import BruteCA

# ConAcq1/ConAcq2 depend on python-sat (pysat), which is an optional dependency
# (it is not needed by QuAcq/MQuAcq/MQuAcq2/GrowAcq/BruteCA). Import them lazily
# so the package still works when pysat is unavailable.
try:
    from .conacq2 import ConAcq2
    from .conacq import ConAcq1
except Exception:  # pragma: no cover - pysat not installed
    ConAcq2 = None
    ConAcq1 = None

