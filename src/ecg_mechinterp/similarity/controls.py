"""Controls every similarity/subspace number in this package should be read against:

- **Null**: the same measure after permuting sample order in one representation (or shuffling
  tokens) — the floor from dimensionality/sample-size alone.
- **Ceiling**: the same encoder and layer compared across two disjoint halves of the training
  patients — the highest similarity achievable given finite data. Cross-encoder similarity is
  meaningful mainly as a fraction of this.
- **Random-encoder floor**: the same measure between two *architecture-matched,
  random-initialization* encoders. This is the control the earlier PTB-XL multi-model study
  didn't have and the RQ1-4 notebook added: if an untrained pair already reaches, say, CKA
  0.86, then a pretrained cross-model CKA of 0.65 is not evidence of a shared *learned*
  concept — it's below what shared architecture alone predicts. See docs/findings.md.

Without at least the null and one of {ceiling, random floor}, a raw similarity number from
this package should not be quoted in a write-up — this is a standing rule (docs/methodology.md),
not a style preference.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ecg_mechinterp.probing.probe import patient_bootstrap
from ecg_mechinterp.similarity.metrics import SIMILARITY_FUNCS, linear_cka


def subsample_rows(*arrays: np.ndarray, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = min(n, len(arrays[0]))
    sel = rng.choice(len(arrays[0]), m, replace=False)
    return [a[sel] for a in arrays]


def similarity_with_reference(a: np.ndarray, b: np.ndarray, patients: np.ndarray,
                               n_permutation: int = 200, n_bootstrap: int = 100, seed: int = 0) -> Dict[str, float]:
    """Point estimate, permutation null, and patient-bootstrap CI (for linear CKA only, the
    cheapest measure to bootstrap) for every measure in SIMILARITY_FUNCS."""
    out = {}
    for name, fn in SIMILARITY_FUNCS.items():
        out[name] = fn(a, b)
        rng = np.random.default_rng(seed)
        null = [fn(a, b[rng.permutation(len(b))]) for _ in range(max(5, n_permutation // 20))]
        out[name + "_null"] = float(np.mean(null))
    lo, hi = patient_bootstrap(lambda idx: linear_cka(a[idx], b[idx]), patients, n=min(100, n_bootstrap), seed=seed)
    out["cka_linear_lo"], out["cka_linear_hi"] = lo, hi
    return out


def within_encoder_ceiling(activations_half_a: np.ndarray, activations_half_b: np.ndarray,
                            n_subsample: int = 4000, seed: int = 0) -> Dict[str, float]:
    """Similarity of the same encoder/layer against itself, on two disjoint patient halves —
    the noise ceiling every cross-encoder number should be divided by before being called
    "high" or "low"."""
    a, b = subsample_rows(activations_half_a, activations_half_b, n=n_subsample, seed=seed)
    return {name: fn(a, b) for name, fn in SIMILARITY_FUNCS.items()}
