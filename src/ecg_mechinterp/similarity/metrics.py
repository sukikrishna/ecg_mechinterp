"""Representation similarity measures, each sensitive to a different aspect of geometry:

| measure       | invariant to                          | sensitive to                              |
|---------------|----------------------------------------|--------------------------------------------|
| linear CKA    | orthogonal transform, isotropic scale  | overall geometry, dominated by high-var dirs|
| RBF CKA       | orthogonal transform, isotropic scale  | local structure via a kernel bandwidth      |
| Procrustes    | orthogonal transform, global scale     | full geometry incl. low-variance directions |
| SVCCA         | invertible linear transform            | shared linear subspace, ignores variance    |
| kNN overlap   | any smooth monotone reparameterization | local neighborhood structure                |

Agreement across all five is real evidence of a shared representation. Agreement on CKA alone
is not, because CKA saturates on the top few principal directions — see docs/methodology.md's
standing rule to always report a null and a within-encoder ceiling alongside any of these
(similarity/controls.py), not the raw number by itself.

Ported from the RQ1-4 stability notebook, which is the only one of the three source analyses
to implement more than linear CKA — this is deliberately the more complete module.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes


def _center(a: np.ndarray) -> np.ndarray:
    return a - a.mean(0, keepdims=True)


def linear_cka(a: np.ndarray, b: np.ndarray) -> float:
    """Linear Centered Kernel Alignment (Kornblith et al., 2019). 0 = unrelated, 1 = identical
    up to an orthogonal transform and isotropic scaling."""
    a, b = _center(np.asarray(a, np.float64)), _center(np.asarray(b, np.float64))
    hsic = np.linalg.norm(b.T @ a, "fro") ** 2
    na, nb = np.linalg.norm(a.T @ a, "fro"), np.linalg.norm(b.T @ b, "fro")
    return float(hsic / (na * nb + 1e-12))


def rbf_cka(a: np.ndarray, b: np.ndarray, sigma_frac: float = 0.5, max_n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = min(max_n, len(a))
    sel = rng.choice(len(a), n, replace=False)

    def gram(m):
        m = _center(np.asarray(m[sel], np.float64))
        sq = np.sum(m ** 2, 1)
        d2 = sq[:, None] + sq[None, :] - 2 * m @ m.T
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        k = np.exp(-d2 / (2 * (sigma_frac ** 2) * med + 1e-12))
        h = np.eye(n) - 1.0 / n
        return h @ k @ h

    ka, kb = gram(a), gram(b)
    return float((ka * kb).sum() / (np.sqrt((ka * ka).sum() * (kb * kb).sum()) + 1e-12))


def procrustes_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 minus normalized Procrustes distance. 1 means identical up to rotation and scale."""
    a, b = _center(np.asarray(a, np.float64)), _center(np.asarray(b, np.float64))
    d = max(a.shape[1], b.shape[1])
    a = np.pad(a, ((0, 0), (0, d - a.shape[1])))
    b = np.pad(b, ((0, 0), (0, d - b.shape[1])))
    a /= np.linalg.norm(a, "fro") + 1e-12
    b /= np.linalg.norm(b, "fro") + 1e-12
    r, _ = orthogonal_procrustes(a, b)
    return float(1.0 - 0.5 * np.linalg.norm(a @ r - b, "fro") ** 2)


def svcca(a: np.ndarray, b: np.ndarray, var_keep: float = 0.99, top_k: int = None) -> float:
    def reduce(m):
        m = _center(np.asarray(m, np.float64))
        u, s, _ = np.linalg.svd(m, full_matrices=False)
        c = np.cumsum(s ** 2) / max(1e-12, np.sum(s ** 2))
        k = int(np.searchsorted(c, var_keep) + 1)
        k = max(2, min(k, m.shape[1], m.shape[0] - 1))
        return u[:, :k] * s[:k]

    ar, br = reduce(a), reduce(b)
    qa, _ = np.linalg.qr(ar)
    qb, _ = np.linalg.qr(br)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    k = top_k or len(s)
    return float(np.mean(np.clip(s[:k], 0, 1)))


def knn_overlap(a: np.ndarray, b: np.ndarray, k: int = 20, max_n: int = 2000, seed: int = 0) -> float:
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    n = min(max_n, len(a))
    sel = rng.choice(len(a), n, replace=False)
    na = NearestNeighbors(n_neighbors=k + 1).fit(a[sel])
    nb = NearestNeighbors(n_neighbors=k + 1).fit(b[sel])
    ia = na.kneighbors(a[sel], return_distance=False)[:, 1:]
    ib = nb.kneighbors(b[sel], return_distance=False)[:, 1:]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ia, ib)]))


SIMILARITY_FUNCS = {
    "cka_linear": linear_cka,
    "cka_rbf": rbf_cka,
    "procrustes": procrustes_similarity,
    "svcca": svcca,
    "knn_overlap": knn_overlap,
}
