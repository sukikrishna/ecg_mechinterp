"""Window -> record -> patient score aggregation.

Probes train on window-level activations but must be *scored* after aggregating to the
patient level — scoring at the window level reports an effective n of "however many windows",
when the real n is the patient count, often an order of magnitude smaller. Ported and
generalized from the CLEF SAE convergence study's evaluation.py (which was MI-vs-HC specific)
to any binary concept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def patient_scores(window_prob: np.ndarray, record: np.ndarray, patient: np.ndarray) -> pd.Series:
    """Window-level probabilities -> one score per patient, via record-level then
    patient-level averaging (so a patient with more windows/records doesn't dominate)."""
    d = pd.DataFrame({"p": window_prob, "rec": record, "patient": patient})
    per_record = d.groupby(["patient", "rec"], observed=True)["p"].mean().reset_index()
    return per_record.groupby("patient", observed=True)["p"].mean()


def patient_labels(patient: np.ndarray, y: np.ndarray) -> pd.Series:
    return pd.DataFrame({"patient": patient, "y": y}).groupby("patient")["y"].first()


def aggregate_auroc(window_prob: np.ndarray, record: np.ndarray, patient: np.ndarray, y: np.ndarray) -> float:
    """Patient-level AUROC for one fold's test windows."""
    scores = patient_scores(window_prob, record, patient)
    labels = patient_labels(patient, y).loc[scores.index]
    if labels.nunique() < 2:
        return float("nan")
    return roc_auc_score(labels.values, scores.values)
