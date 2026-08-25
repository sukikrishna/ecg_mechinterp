"""Pure-math correctness checks for similarity/subspace metrics — no data download or model
weights needed. Run with `pytest tests/`."""
import numpy as np

from ecg_mechinterp.similarity.metrics import knn_overlap, linear_cka, procrustes_similarity, svcca
from ecg_mechinterp.similarity.subspace import (
    pca_subspace,
    principal_angles,
    stable_rank,
    subspace_overlap,
    subspace_scores,
)


def _rng():
    return np.random.default_rng(0)


def test_cka_invariant_to_orthogonal_transform():
    rng = _rng()
    a = rng.standard_normal((500, 40))
    r = np.linalg.qr(rng.standard_normal((40, 40)))[0]
    b = a @ r
    assert linear_cka(a, b) > 0.999
    assert procrustes_similarity(a, b) > 0.999
    assert svcca(a, b) > 0.999
    assert knn_overlap(a, b, k=10) > 0.99


def test_cka_low_for_unrelated_data():
    rng = _rng()
    a = rng.standard_normal((500, 40))
    c = rng.standard_normal((500, 40))
    assert linear_cka(a, c) < 0.2


def test_principal_angles_small_for_shared_subspace():
    rng = _rng()
    n, d, r = 2000, 40, 5
    true_basis = np.linalg.qr(rng.standard_normal((d, r)))[0].T
    scores = rng.standard_normal((n, r))
    signal = scores @ true_basis
    a = signal + 0.05 * rng.standard_normal((n, d))
    b = signal + 0.05 * rng.standard_normal((n, d))
    angles = np.degrees(principal_angles(pca_subspace(a, r), pca_subspace(b, r)))
    assert angles.max() < 5.0


def test_principal_angles_large_for_independent_subspaces():
    rng = _rng()
    d, r = 40, 5
    basis_a = np.linalg.qr(rng.standard_normal((d, r)))[0].T
    basis_b = np.linalg.qr(rng.standard_normal((d, r)))[0].T
    angles = np.degrees(principal_angles(basis_a, basis_b))
    assert angles.min() > 30.0


def test_subspace_overlap_and_stable_rank():
    rng = _rng()
    a = rng.standard_normal((500, 40))
    r = np.linalg.qr(rng.standard_normal((40, 40)))[0]
    b = a @ r
    overlap = subspace_overlap(subspace_scores(a, 8), subspace_scores(b, 8))
    assert overlap > 0.99
    assert stable_rank({1: 0.9, 2: 0.85, 4: 0.75, 8: 0.5}, threshold=0.7) == 4
    assert stable_rank({1: 0.5}, threshold=0.7) == 0
