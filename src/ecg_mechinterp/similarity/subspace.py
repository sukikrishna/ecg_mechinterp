"""Subspace-level comparison: principal angles, subspace overlap, and stable rank.

The core tool for RQ3 (docs/findings.md): individual directions can be unstable while the
subspace they span is reproducible, or vice versa — several results across all three source
analyses only make sense read at this level, not the individual-direction level.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def pca_subspace(activations: np.ndarray, n_components: int) -> np.ndarray:
    """Orthonormal basis (n_components, n_features) spanning the top-variance subspace."""
    pca = PCA(n_components=n_components)
    pca.fit(activations)
    return pca.components_


def principal_angles(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    """Principal angles (radians, ascending) between two subspaces given as orthonormal bases
    of shape (k, n_features). 0 means the subspaces overlap exactly in that direction; pi/2
    means fully orthogonal in that direction."""
    cosines = np.linalg.svd(basis_a @ basis_b.T, compute_uv=False)
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def decoder_cosine_similarity(decoder_a: np.ndarray, decoder_b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between two sets of vectors (e.g. SAE decoder columns),
    shape (n_a, dim) and (n_b, dim) — used to match features across seeds/runs."""
    a = decoder_a / np.linalg.norm(decoder_a, axis=1, keepdims=True)
    b = decoder_b / np.linalg.norm(decoder_b, axis=1, keepdims=True)
    return a @ b.T


def subspace_scores(z_tok: np.ndarray, k: int) -> np.ndarray:
    """Project centered tokens onto their own top-k principal directions; (n, k) scores."""
    z = z_tok - z_tok.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    k = min(k, vt.shape[0])
    return z @ vt[:k].T


def subspace_overlap(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Mean squared cosine of principal angles between the column spaces of two score
    matrices — 1.0 means the two subspaces coincide, and the expected value under
    independence falls with rank, which is why this is always read against a shuffled-token
    null at the same rank (see similarity/controls.py), not against 0."""
    qa, _ = np.linalg.qr(scores_a - scores_a.mean(0, keepdims=True))
    qb, _ = np.linalg.qr(scores_b - scores_b.mean(0, keepdims=True))
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def stable_rank(overlap_by_rank: Dict[int, float], threshold: float = 0.7) -> int:
    """Largest rank k at which `overlap_by_rank[k] >= threshold` — "how many dimensions are
    shared", read alongside the within-encoder ceiling at the same ranks, not in isolation."""
    ok = [k for k, v in overlap_by_rank.items() if v >= threshold]
    return max(ok) if ok else 0


def stable_rank_table(rows: list, threshold: float = 0.7) -> pd.DataFrame:
    """`rows`: list of dicts with at least {kind, a, b, rank, overlap} (one row per rank, as
    produced by scripts/run_stability_suite.py). Returns one stable-rank row per (kind, a, b)."""
    df = pd.DataFrame(rows)
    out = []
    for (kind, a, b), grp in df.groupby(["kind", "a", "b"]):
        ok = grp[grp.overlap >= threshold]["rank"]
        out.append(dict(kind=kind, a=a, b=b, stable_rank=int(ok.max()) if len(ok) else 0))
    return pd.DataFrame(out).sort_values(["kind", "stable_rank"], ascending=[True, False])
