"""Dataset-agnostic loading entry point for scripts/: dispatches on `cfg.dataset` and returns
a common shape (`X`, `win_meta`) so every downstream script (probing, similarity, SAE, causal)
is written once against that shape rather than once per dataset.

- `X`: (n_windows, samples, leads) float32 raw waveform, one row per window, native sampling
  rate for the dataset (500Hz/10s for PTB-DB after windowing, PTB-XL's own per-record rate).
  Each model's own `preprocess()` still does its model-specific resampling/filtering/z-score
  on top of this — this loader never applies a model-specific transform.
- `win_meta`: one row per window with at least {patient, record, dx_group, sex, age,
  sex_male}, everything downstream (splits, concept masks, demographic checks) needs.

PTB-DB windowing follows `data/windows.py`; PTB-XL records are already fixed-length so no
windowing step is needed there — each record is one "window".
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import wfdb
from tqdm.auto import tqdm

from ecg_mechinterp.config import Config
from ecg_mechinterp.data import ptbdb, ptbxl
from ecg_mechinterp.data.windows import preprocess_signal, quality_filter, window_signal
from ecg_mechinterp.utils import DiskCache


def load_ptbdb_windows(cfg: Config) -> Tuple[np.ndarray, pd.DataFrame]:
    cache = DiskCache(cfg.cache_dir)
    cache_name = f"ptbdb_windows_lead{cfg.lead}.npz"
    if cache.exists(cache_name):
        z = np.load(cache.path(cache_name), allow_pickle=True)
        return z["X"], pd.DataFrame(z["meta"], columns=z["meta_cols"])

    root = Path(cfg.data_dir) / "raw" / "ptbdb"
    local_dir = str(root) if root.exists() else None
    hea_files = sorted(root.rglob("*.hea")) if local_dir else []
    if not hea_files:
        raise FileNotFoundError(
            f"no PTB-DB data found under {root}; run scripts/download_ptbdb.sh first"
        )
    records = []
    for hea in hea_files:
        rec_id = str(hea.relative_to(root)).removesuffix(".hea")
        header = wfdb.rdheader(str(hea)[: -len(".hea")])
        comments = ptbdb.parse_comments(header.comments)
        records.append(dict(record=rec_id, patient=Path(rec_id).parent.name,
                            reason=comments.get("reason for admission", "n/a"),
                            sex=comments.get("sex"), age=comments.get("age")))
    meta = ptbdb.build_metadata(pd.DataFrame(records))
    if cfg.max_records:
        meta = meta.iloc[: cfg.max_records]

    all_windows, rows = [], []
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc="ptbdb windows"):
        sig, fs = ptbdb.read_signal(r["record"], cfg.lead, local_dir=local_dir)
        if sig is None:
            continue
        proc = preprocess_signal(sig, fs)
        try:
            wins = window_signal(proc, fs, cfg.window_sec, cfg.window_len, cfg.max_windows_per_record)
        except ValueError:
            continue
        keep = quality_filter(wins)
        wins = wins[keep]
        if wins.shape[0] == 0:
            continue
        all_windows.append(wins)
        for j in range(wins.shape[0]):
            rows.append([r["record"], r["patient"], j, r["dx_group"], r["sex"], r["age"]])

    x = np.concatenate(all_windows, 0).astype(np.float32)[..., None]  # (n, samples, 1 lead)
    win_meta = pd.DataFrame(rows, columns=["record", "patient", "win_idx", "dx_group", "sex", "age"])
    win_meta["age"] = pd.to_numeric(win_meta["age"], errors="coerce")
    win_meta["sex_male"] = win_meta["sex"].map({"male": 1, "female": 0})
    win_meta["is_mi"] = (win_meta["dx_group"] == "MI").astype(int)

    np.savez_compressed(cache.path(cache_name), X=x,
                        meta=win_meta.values.astype(object), meta_cols=np.array(win_meta.columns))
    return x, win_meta


def load_ptbxl_windows(cfg: Config) -> Tuple[np.ndarray, pd.DataFrame]:
    root = Path(cfg.data_dir) / "raw" / "ptb-xl"
    db = ptbxl.PTBXL.load(root)
    labels = db.concept_labels()
    demo = db.demographics()
    ecg_ids = labels.index[: cfg.max_records] if cfg.max_records else labels.index
    sampling_rate = int(cfg.target_fs) if cfg.target_fs in (100, 500) else 100
    signals = db.load_waveforms(ecg_ids, sampling_rate=sampling_rate)  # (n, samples, 12)

    win_meta = pd.DataFrame({"record": ecg_ids, "patient": ecg_ids}, index=ecg_ids)
    for concept in labels.columns:
        win_meta[concept] = labels.loc[ecg_ids, concept].values
    win_meta["dx_group"] = np.where(win_meta["myocardial_infarction"], "MI",
                                    np.where(win_meta["normal_rhythm"], "NORM", "OTHER"))
    win_meta["sex"] = demo.loc[ecg_ids, "sex"].map({0: "male", 1: "female"}).values
    win_meta["age"] = demo.loc[ecg_ids, "age"].values
    win_meta["sex_male"] = win_meta["sex"].map({"male": 1, "female": 0})
    win_meta = win_meta.reset_index(drop=True)
    return signals.astype(np.float32), win_meta


def load_windows(cfg: Config) -> Tuple[np.ndarray, pd.DataFrame]:
    if cfg.dataset == "ptbdb":
        return load_ptbdb_windows(cfg)
    if cfg.dataset == "ptbxl":
        return load_ptbxl_windows(cfg)
    raise ValueError(f"unknown dataset {cfg.dataset!r}; expected 'ptbxl' or 'ptbdb'")
