"""Probing/splits/evaluation checks on synthetic patient-structured data."""
import numpy as np
import pandas as pd

from ecg_mechinterp.data.splits import held_out_split, make_patient_splits
from ecg_mechinterp.evaluation import aggregate_auroc
from ecg_mechinterp.probing.probe import fit_probe, probe_direction_stability


def _synthetic_patients(n_patients=60, windows_per_patient=10, d=20, seed=0):
    rng = np.random.default_rng(seed)
    patients = np.repeat(np.arange(n_patients), windows_per_patient)
    y_patient = np.arange(n_patients) % 2
    y = y_patient[patients]
    signal_dir = rng.standard_normal(d)
    z = y[:, None] * signal_dir[None, :] * 2 + rng.standard_normal((len(patients), d))
    return patients, y, z, signal_dir


def test_fit_probe_recovers_clean_signal():
    patients, y, z, _ = _synthetic_patients()
    train_pat, test_pat = np.arange(0, 40), np.arange(40, 60)
    tr, te = np.isin(patients, train_pat), np.isin(patients, test_pat)
    res = fit_probe(z[tr], y[tr], z[te], y[te], patients[te], kind="binary", n_bootstrap=100, seed=0)
    assert res["value"] > 0.85
    assert res["lo"] <= res["value"] <= res["hi"]


def test_probe_direction_stability_high_for_single_direction_signal():
    patients, y, z, _ = _synthetic_patients()
    pds = probe_direction_stability(z, y, patients, kind="binary", n_boot=15, seed=0)
    assert pds > 0.8


def test_make_patient_splits_no_patient_leakage():
    wm = pd.DataFrame({
        "patient": np.repeat(np.arange(60), 10),
        "dx_group": np.tile(np.repeat(["MI", "HC"], 5), 60),
        "sex": np.tile(["male", "female"], 300),
    })
    splits = make_patient_splits(wm, seed=0)
    train_patients = set(wm.loc[splits["train"], "patient"])
    test_patients = set(wm.loc[splits["test"], "patient"])
    val_patients = set(wm.loc[splits["val"], "patient"])
    assert not (train_patients & test_patients)
    assert not (train_patients & val_patients)
    assert not (test_patients & val_patients)


def test_held_out_split_no_leakage():
    wm = pd.DataFrame({"patient": np.repeat(np.arange(40), 5), "dx_group": np.tile(["MI", "HC"], 100)})
    splits = held_out_split(wm, stratify_col="dx_group", seed=0)
    train_patients = set(wm.loc[splits["train"], "patient"])
    held_patients = set(wm.loc[splits["held_out"], "patient"])
    assert not (train_patients & held_patients)


def test_aggregate_auroc_patient_level():
    patients, y, z, signal_dir = _synthetic_patients()
    prob = 1 / (1 + np.exp(-(z @ signal_dir)))
    auroc = aggregate_auroc(prob, patients, patients, y)
    assert auroc > 0.85
