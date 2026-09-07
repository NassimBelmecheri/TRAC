"""
Neural oracle wrapper (T-ORACLE as a CA oracle).

Adapts a trained :class:`CSPAttentionModel` to the :class:`pycona.Oracle`
interface so it can drive the symbolic learner (FASTCA / others) in place of a
human.

Two correctness fixes over the legacy ``TransformerOracle`` in ``TRAC_test.py``:

1. **Positional alignment.** A membership query ``Y`` is a *subset* of variables.
   The legacy code packed the queried values into a sequence positionally
   (``[v.value() for v in query]``), which both misaligns them with the model's
   positional embeddings and breaks when ``len(Y) != n``. We instead build the
   full length-``n`` vector and place each value at *its own variable index*,
   leaving unassigned variables as 0 - exactly the encoding used at training time.

2. **Interface signature.** ``answer_generalization_query`` now matches the
   abstract base (``(self, C)``); the legacy override used ``(self, c, C)``.
"""
import numpy as np
import torch

from . import utils  # noqa: F401  (bootstraps pycona onto sys.path)
from pycona.answering_queries.oracle import Oracle
from .checkpoint import load_checkpoint


class TransformerOracle(Oracle):
    """A trained T-ORACLE exposed as a CA membership oracle."""

    def __init__(self, model, variables, config=None, device="cpu", threshold=0.5):
        super().__init__()
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
        self.config = config or {}

        # Ordered full variable list -> position index (variable identity by name).
        self.variables = list(variables)
        self.n = len(self.variables)
        self.index = {v.name: i for i, v in enumerate(self.variables)}

    # ---- construction ---------------------------------------------------- #
    @classmethod
    def from_checkpoint(cls, path_pth, variables, device=None, threshold=0.5):
        device = device or utils.get_device()
        model, config = load_checkpoint(path_pth, device=device)
        n = len(list(variables))
        if config.get("num_positions") not in (None, n):
            raise ValueError(
                f"Checkpoint expects {config['num_positions']} variables but the "
                f"benchmark has {n}. Was the model trained on this benchmark/scale?"
            )
        return cls(model, variables, config=config, device=device, threshold=threshold)

    # ---- encoding -------------------------------------------------------- #
    def _encode(self, Y):
        """Build the length-n input vector for a (partial) assignment ``Y``."""
        vec = np.zeros(self.n, dtype=np.int64)
        for v in Y:
            idx = self.index.get(getattr(v, "name", None))
            if idx is None:
                continue
            val = v.value() if hasattr(v, "value") else v
            if val is not None:
                vec[idx] = int(val)
        return vec

    @torch.no_grad()
    def _probability(self, vec):
        t = torch.from_numpy(vec).unsqueeze(0).to(self.device)
        mask = (t == 0)
        preds, _, _ = self.model(t, None, src_key_padding_mask=mask)
        return float(preds.squeeze().item())

    # ---- Oracle interface ------------------------------------------------ #
    def answer_membership_query(self, Y):
        if Y is None:
            Y = self.variables
        return self._probability(self._encode(Y)) >= self.threshold

    def answer_recommendation_query(self, c):
        # A neural oracle cannot natively answer recommendation queries.
        return False

    def answer_generalization_query(self, C):
        # A neural oracle cannot natively answer generalization queries.
        return False
