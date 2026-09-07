"""
T-ORACLE: the Transformer-based neural oracle of the TRAC framework.

This is the single, unified implementation of the neural oracle architecture
described in the paper "Learning Symbolic Constraint Representations from
Examples: A Neuro-Symbolic Approach".  The three paper variants (TO1, TO2, TO3)
share *exactly* this architecture and differ only in their training data
(see :mod:`trac.datagen`); the original code base duplicated this file three
times (``TO1/model_1.py``, ``TO2/model_1.py``, ``TO3/model_1.py``).

Module / paper correspondence:
    * Value + Positional embedding .......... eq. (1) & (2)  -> ``value_embed`` + ``pos_embed``
    * Cross-Variable Attention matrix M ..... learned ``constraint_graph_logits``
    * Value Interaction matrix V ............ eq. (3)        -> ``ConstraintGraphComponent``
    * Dependency Score S .................... eq. (4)
    * Gated classification output ........... eq. (5)        -> ``p * sigma(tau - S)``

The class names and parameter names are kept identical to the legacy
``model_1.py`` so that previously trained ``.pth`` checkpoints remain loadable.
"""
import math

import torch
import torch.nn as nn


class ConstraintGraphComponent(nn.Module):
    """Learns a global variable-dependency matrix M and a dynamic value
    interaction matrix V, and combines them into per-variable conflict features
    and a global dependency score S (eq. 3-4)."""

    def __init__(self, num_vars, embed_dim):
        super().__init__()
        # Global structural dependency matrix M (n x n), learned end-to-end.
        # Initialised small to encourage sparsity.
        self.constraint_graph_logits = nn.Parameter(torch.randn(num_vars, num_vars) * 0.01)

        # Query / Key projections for the value-interaction matrix V (eq. 3).
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x_emb, mask):
        """
        :param x_emb: (B, N, D) value+position embeddings
        :param mask:  (B, N) 1 for active (assigned) variables, 0 for padding
        :return: (features (B,N), score S (B,1), matrix A (N,N))
        """
        # 1. Static structure M -> A in [0, 1]
        A = torch.sigmoid(self.constraint_graph_logits)
        A_batch = A.unsqueeze(0)

        # 2. Dynamic value incompatibility V (eq. 3)
        Q = self.query_proj(x_emb)
        K = self.key_proj(x_emb)
        val_interaction = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        V = torch.sigmoid(val_interaction)

        # 3. Conflict exists only if a structural dependency (A) AND a value clash (V)
        C = A_batch * V

        # 4. Mask padding pairs and self-loops
        active_pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
        C = C * active_pair_mask
        eye = torch.eye(C.size(1), device=C.device).unsqueeze(0)
        C = C * (1 - eye)

        # 5. Per-variable conflict features
        features = C.sum(dim=2)  # (B, N)

        # 6. Global dependency score S (eq. 4): mean conflict over active pairs
        total_conflict_sum = C.sum(dim=(1, 2)).unsqueeze(1)  # (B, 1)
        num_active = mask.sum(dim=1).unsqueeze(1)
        num_pairs = torch.clamp(num_active * (num_active - 1), min=1.0)
        S = total_conflict_sum / num_pairs

        return features, S, A


class CSPAttentionModel(nn.Module):
    """The T-ORACLE network.

    :param n_vars: maximum domain value ``d_max`` (vocabulary size; value 0 is
        reserved for unassigned / padding). NOTE: kept as the first positional
        argument for checkpoint compatibility with the legacy code.
    :param grid_size: number of variables ``n`` (sequence length).
    """

    def __init__(self, n_vars, grid_size, embed_dim=32, num_heads=4):
        super().__init__()

        # Embeddings (eq. 1 & 2)
        self.value_embed = nn.Embedding(n_vars + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, grid_size, embed_dim))

        # Transformer encoder (global receptive field over the assignment)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Constraint-graph component (M, V, S)
        self.conflict_net = ConstraintGraphComponent(grid_size, embed_dim)

        # Learnable gating threshold tau (eq. 5)
        self.tau = nn.Parameter(torch.tensor(0.5))

        # MLP classifier over [transformer features ; conflict features]
        classifier_input_dim = (grid_size * embed_dim) + grid_size
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward(self, x, y=None, src_key_padding_mask=None):
        x = x.long()
        val_emb = self.value_embed(x)
        x_emb = val_emb + self.pos_embed

        if src_key_padding_mask is None:
            active_mask = (x != 0).float()
            transformer_mask = (x == 0)
        else:
            active_mask = (~src_key_padding_mask).float()
            transformer_mask = src_key_padding_mask

        # 1. Transformer path
        trans_out = self.transformer(x_emb, src_key_padding_mask=transformer_mask)
        trans_flat = trans_out.flatten(start_dim=1)

        # 2. Conflict path (matrix M / V -> features, S)
        conflict_feats, S, learned_matrix = self.conflict_net(x_emb, active_mask)

        # 3. Base probability p
        combined = torch.cat([trans_flat, conflict_feats], dim=1)
        p = torch.sigmoid(self.classifier(combined))

        # 4. Gating (eq. 5): suppress output when the dependency score S is high
        gate_scale = 10.0
        soft_gate = torch.sigmoid(gate_scale * (self.tau - S))
        final_output = p * soft_gate

        # Return raw logits of M as well so an L1 sparsity penalty can be applied.
        return final_output, self.conflict_net.constraint_graph_logits, learned_matrix


def build_model(num_values: int, num_positions: int, embed_dim: int = 32,
                num_heads: int = 4) -> CSPAttentionModel:
    """Convenience constructor with clearer argument names.

    :param num_values: maximum domain value d_max (vocabulary size).
    :param num_positions: number of variables n (sequence length).
    """
    return CSPAttentionModel(num_values, num_positions, embed_dim=embed_dim, num_heads=num_heads)
