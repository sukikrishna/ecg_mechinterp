"""Cross-seed / cross-encoder SAE stability metrics — the direct test of individual-feature
reproducibility (RQ2 in docs/findings.md).

Two matching procedures, because the right one depends on what's being compared:

- `dictionary_matching` (same encoder+layer, different seed or method): decoder directions
  live in the same space, so cosine similarity is directly meaningful.
- `activation_matching` (different encoders, different dimensionality, or just a
  dimension-free check): match features by the correlation of their activation profiles over
  a shared, aligned token set. The only option across encoders; also usable within an encoder
  so the two procedures can be compared on the same pairs.

Every metric here has a null (`random_dictionary_null`, `permuted_pairs`) — a raw MMCS number
floors well above zero purely from dictionary size, so it is not interpretable without one
(see docs/methodology.md).
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ecg_mechinterp.sae.model import SAE

HUNGARIAN_MAX = 2048  # optimal assignment is cubic; above this it's solved on a random subset


def dictionary_matching(sae_a: SAE, sae_b: SAE, seed: int = 0) -> Dict[str, object]:
    wa = F.normalize(sae_a.W_dec.detach(), dim=1)
    wb = F.normalize(sae_b.W_dec.detach(), dim=1)
    c = (wa @ wb.T).cpu().numpy()
    mmcs_ab = c.max(1)
    if c.shape[0] <= HUNGARIAN_MAX:
        ri, ci = linear_sum_assignment(-c)
        hung = float(c[ri, ci].mean())
    else:
        rng = np.random.default_rng(seed)
        sa = rng.choice(c.shape[0], HUNGARIAN_MAX, replace=False)
        sb = rng.choice(c.shape[1], HUNGARIAN_MAX, replace=False)
        sub = c[np.ix_(sa, sb)]
        ri, ci = linear_sum_assignment(-sub)
        hung = float(sub[ri, ci].mean())
    return dict(mmcs=float(mmcs_ab.mean()), mmcs_median=float(np.median(mmcs_ab)),
                hungarian=hung, per_feature=mmcs_ab)


def random_dictionary_null(d_in: int, d_hidden: int, seed: int = 0, reps: int = 3) -> float:
    """MMCS between two untrained random dictionaries of the given shape — the floor that
    dimensionality/dictionary-size alone predicts, with no learning involved at all."""
    vals = []
    for r in range(reps):
        g = torch.Generator().manual_seed(seed + r)
        a = F.normalize(torch.randn(d_hidden, d_in, generator=g), dim=1)
        b = F.normalize(torch.randn(d_hidden, d_in, generator=g), dim=1)
        vals.append(float((a @ b.T).max(1).values.mean()))
    return float(np.mean(vals))


def activation_matching(z_a: np.ndarray, z_b: np.ndarray, min_density: float = 1e-4,
                         max_tokens: int = 40_000, seed: int = 0) -> Dict[str, object]:
    """Correlation-based feature matching over shared tokens. `z_a`, `z_b`: (n_tokens, F)."""
    rng = np.random.default_rng(seed)
    n = min(max_tokens, len(z_a))
    sel = rng.choice(len(z_a), n, replace=False)
    a, b = z_a[sel], z_b[sel]
    ka, kb = a.mean(0) > min_density, b.mean(0) > min_density
    a, b = a[:, ka], b[:, kb]
    if a.shape[1] < 2 or b.shape[1] < 2:
        return dict(mmcs=np.nan, hungarian=np.nan, per_feature=np.array([]), alive=(int(ka.sum()), int(kb.sum())))
    a = (a - a.mean(0)) / (a.std(0) + 1e-8)
    b = (b - b.mean(0)) / (b.std(0) + 1e-8)
    c = (a.T @ b) / len(a)
    best = c.max(1)
    m = min(c.shape)
    ri, ci = linear_sum_assignment(-c[:m, :m])
    return dict(mmcs=float(best.mean()), hungarian=float(c[:m, :m][ri, ci].mean()), per_feature=best,
                alive=(int(ka.sum()), int(kb.sum())), alive_idx=(np.flatnonzero(ka), np.flatnonzero(kb)))


def permuted_pairs(n_features: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffled correspondence between two equal-size feature sets — the floor for any
    matched-pair metric (e.g. the per-pair Pearson correlation in StreamingPearson below)."""
    rng = np.random.default_rng(seed)
    return np.arange(n_features), rng.permutation(n_features)


class StreamingPearson:
    """Pearson r for a fixed set of matched feature pairs, accumulated in one pass without
    ever materializing the full held-out activation matrix (which can be hundreds of MB at
    realistic token counts and dictionary sizes)."""

    def __init__(self, n_pairs: int):
        z = np.zeros(n_pairs, dtype=np.float64)
        self.n = 0
        self.sa, self.sb = z.copy(), z.copy()
        self.saa, self.sbb, self.sab = z.copy(), z.copy(), z.copy()

    def update(self, a: np.ndarray, b: np.ndarray) -> None:
        """a, b: (batch, n_pairs) matched activations."""
        self.n += a.shape[0]
        self.sa += a.sum(0)
        self.sb += b.sum(0)
        self.saa += (a * a).sum(0)
        self.sbb += (b * b).sum(0)
        self.sab += (a * b).sum(0)

    def result(self) -> np.ndarray:
        n = self.n
        cov = self.sab - self.sa * self.sb / n
        va = self.saa - self.sa ** 2 / n
        vb = self.sbb - self.sb ** 2 / n
        den = np.sqrt(np.clip(va, 0, None) * np.clip(vb, 0, None))
        return np.divide(cov, den, out=np.zeros_like(cov), where=den > 1e-12)
