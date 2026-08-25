"""Linear probing of representations, with patient-level bootstrap confidence intervals.

`patient_bootstrap` is the standing rule from docs/methodology.md applied to uncertainty
estimation specifically: resampling *windows* would understate any interval, often by a large
factor, because windows from one patient are close to identical — the effective n is the
patient count, not the window count.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def linear_probe(activations: np.ndarray, labels: np.ndarray, test_size: float = 0.2,
                  seed: int = 0) -> dict:
    """Train a logistic-regression probe on flattened `activations` to predict binary
    `labels`, with a single random split (no patient grouping) — use `fit_probe` below
    instead whenever windows are grouped by patient, which is true of every dataset this
    package loads. Kept for quick, ungrouped sanity checks only.

    `direction` is returned in the *standardized* feature space; it is what gets compared for
    stability across seeds/models (cosine similarity, principal angles) elsewhere.
    """
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        activations, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    probe = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    probe.fit(x_train, y_train)
    scores = probe.predict_proba(x_test)[:, 1]
    return {
        "direction": probe.coef_[0],
        "bias": probe.intercept_[0],
        "auc": roc_auc_score(y_test, scores),
        "accuracy": probe.score(x_test, y_test),
    }


def patient_bootstrap(metric_fn: Callable[[np.ndarray], float], patients: np.ndarray,
                       n: int = 500, seed: int = 0) -> Tuple[float, float]:
    """95% CI for `metric_fn(idx)` (idx an array of row indices) via patient-level resampling."""
    uniq = np.unique(patients)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(patients == p) for p in draw])
        try:
            vals.append(metric_fn(idx))
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def fit_probe(z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray,
              patients_test: np.ndarray, kind: str = "binary", C: float = 1.0,
              n_bootstrap: int = 500, seed: int = 0) -> dict:
    """Fit on a pre-split train set, score on test with a patient-bootstrap CI. `kind` is
    "binary" (AUROC via logistic regression) or "continuous" (Spearman rho via ridge
    regression, for a concept like age)."""
    if kind == "binary":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=seed))
        clf.fit(z_train, y_train)
        scores = clf.decision_function(z_test)
        point = roc_auc_score(y_test, scores)
        lo, hi = patient_bootstrap(
            lambda idx: roc_auc_score(y_test[idx], scores[idx]) if len(np.unique(y_test[idx])) > 1 else np.nan,
            patients_test, n=n_bootstrap, seed=seed,
        )
        return dict(metric="AUROC", value=point, lo=lo, hi=hi, model=clf, score=scores)
    clf = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    clf.fit(z_train, y_train)
    scores = clf.predict(z_test)
    point = sps.spearmanr(y_test, scores).statistic
    lo, hi = patient_bootstrap(
        lambda idx: sps.spearmanr(y_test[idx], scores[idx]).statistic, patients_test, n=n_bootstrap, seed=seed,
    )
    return dict(metric="Spearman", value=point, lo=lo, hi=hi, model=clf, score=scores)


def probe_direction_stability(z: np.ndarray, y: np.ndarray, patients: np.ndarray, kind: str = "binary",
                               n_boot: int = 20, seed: int = 0) -> float:
    """Mean pairwise cosine similarity between probe weight vectors refit on bootstrap
    resamples of patients. 1.0 means the concept is a single reproducible direction; values
    near 0 mean the direction wanders across resamples even though the probe's AUROC/rho may
    look stable — see docs/findings.md's RQ3 discussion of why this is a different, and
    usually lower, number than subspace-level stability."""
    uniq = np.unique(patients)
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(n_boot):
        draw = rng.choice(uniq, len(uniq), replace=True)
        sel = np.concatenate([np.flatnonzero(patients == p) for p in draw])
        if kind == "binary" and len(np.unique(y[sel])) < 2:
            continue
        est = LogisticRegression(C=1.0, max_iter=2000) if kind == "binary" else Ridge(alpha=1.0)
        scaler = StandardScaler().fit(z[sel])
        est.fit(scaler.transform(z[sel]), y[sel])
        w = np.ravel(est.coef_)
        weights.append(w / (np.linalg.norm(w) + 1e-12))
    if len(weights) < 2:
        return float("nan")
    w_stack = np.stack(weights)
    cos = w_stack @ w_stack.T
    iu = np.triu_indices(len(w_stack), 1)
    return float(np.mean(cos[iu]))


def dims_needed(z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray,
                 kind: str = "binary", ks: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128)) -> Dict[int, float]:
    """AUROC (binary) or Spearman rho (continuous) using only the top-k PCA directions of the
    training activations, for each k — separates "one stable direction carries the concept"
    from "a k-dimensional stable subspace carries it"."""
    mu = z_train.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(z_train - mu, full_matrices=False)
    out = {}
    for k in ks:
        if k > vt.shape[0]:
            continue
        p = vt[:k].T
        ztr, zte = (z_train - mu) @ p, (z_test - mu) @ p
        if kind == "binary":
            if len(np.unique(y_train)) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            clf.fit(ztr, y_train)
            out[k] = round(float(roc_auc_score(y_test, clf.decision_function(zte))), 3)
        else:
            clf = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            clf.fit(ztr, y_train)
            out[k] = round(float(sps.spearmanr(y_test, clf.predict(zte)).statistic), 3)
    return out
