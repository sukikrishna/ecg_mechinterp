"""PTB Diagnostic ECG Database (PhysioNet `ptbdb` v1.0.0) loading utilities.

Merges two acquisition paths that were each built independently and validated against real
data (see docs/provenance.md):

- `read_signal`/`read_header` (from the RQ1-4 stability notebook): reads via `wfdb`, either
  from a local copy in `cfg.data_dir` or streamed directly from PhysioNet (`pn_dir=`) with no
  download step at all.
- `ZipReader` (from the CLEF SAE convergence study): reads one record at a time out of the
  PhysioNet distribution zip without ever extracting it — useful when disk space is the
  binding constraint (the original use case: 5.6 GB free against a ~15 GB extracted size).

Both are kept because they solve different constraints; pick whichever fits the machine
this runs on. `DX_GROUPS`/`map_group` (free-text "reason for admission" -> diagnostic group)
and `cohort_demographics` (the age/sex-vs-diagnosis confound check) are the notebook's, used
as-is since PTB-DB's demographic skew is exactly what motivates checking PTB-XL the same way
(see docs/findings.md).
"""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import wfdb
from scipy import stats as sps

PTBDB_PN_DIR = "ptbdb"

DX_GROUPS = {
    "myocardial infarction": "MI",
    "healthy control": "HC",
    "cardiomyopathy": "CM",
    "heart failure (nyha 2)": "CM",
    "heart failure (nyha 3)": "CM",
    "heart failure (nyha 4)": "CM",
    "bundle branch block": "BBB",
    "dysrhythmia": "DYS",
    "myocardial hypertrophy": "HYP",
    "valvular heart disease": "VHD",
    "myocarditis": "MYO",
    "hypertrophy": "HYP",
    "stable angina": "ANG",
    "unstable angina": "ANG",
    "palpitation": "MISC",
    "n/a": None,
}


def map_group(reason) -> Optional[str]:
    if not isinstance(reason, str):
        return None
    r = reason.strip().lower()
    if r in DX_GROUPS:
        return DX_GROUPS[r]
    for key, val in DX_GROUPS.items():
        if key in r:
            return val
    return "MISC"


def parse_comments(comments: Sequence[str]) -> Dict[str, str]:
    """PTB-DB headers store a free-text clinical summary as 'key: value' comment lines."""
    out = {}
    for line in comments:
        line = line.strip().lstrip("#").strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip().lower(), val.strip()
        if key and val and val.lower() not in {"n/a", "na", "unknown", "-"}:
            out[key] = val
    return out


def read_header(rec_id: str, local_dir: Optional[str] = None):
    if local_dir:
        return wfdb.rdheader(str(Path(local_dir) / rec_id))
    return wfdb.rdheader(rec_id, pn_dir=PTBDB_PN_DIR)


def read_signal(rec_id: str, channel_name: str, local_dir: Optional[str] = None):
    """Return (signal, fs) for one named channel, e.g. channel_name='i' or 'v5'."""
    if local_dir:
        rec = wfdb.rdrecord(str(Path(local_dir) / rec_id))
    else:
        rec = wfdb.rdrecord(rec_id, pn_dir=PTBDB_PN_DIR)
    names = [n.lower() for n in rec.sig_name]
    if channel_name.lower() not in names:
        return None, rec.fs
    idx = names.index(channel_name.lower())
    return rec.p_signal[:, idx].astype(np.float32), rec.fs


class ZipReader:
    """Reads one record at a time out of the ptbdb distribution zip; never extracts it.

    `lead`: the original CLEF SAE convergence study found lead 'i' failed its own AUROC gate
    (0.889 vs. v5's 0.924) and switched by explicit decision after diagnosing why (see
    docs/provenance.md) — that was specific to CLEF-medium's stage-3 activations on this
    cohort, not a general claim that v5 is the better lead. Default to 'i' here (the
    wearable-relevant lead the stability notebook also uses) and treat 'v5' as a
    documented fallback if a downstream AUROC gate fails on it.
    """

    def __init__(self, zip_path: str, lead: str = "i"):
        self.zf = zipfile.ZipFile(zip_path)
        self.lead = lead

    def read_lead(self, dat_member: str) -> np.ndarray:
        """dat_member: 'ptb-.../patientNNN/sXXXX.dat' -> (n_samples,) float64."""
        hea_member = dat_member[:-4] + ".hea"
        stem = Path(dat_member).stem
        text = self.zf.read(hea_member).decode("latin-1")
        sig_lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")][1:]
        names = [l.split()[8] for l in sig_lines]
        if self.lead not in names:
            raise ValueError(f"{stem}: lead {self.lead!r} not in {names}")
        idx = names.index(self.lead)
        if not sig_lines[idx].split()[0].endswith(".dat"):
            raise ValueError(f"{stem}: lead {self.lead!r} is not stored in the .dat file")
        with tempfile.TemporaryDirectory() as td:
            for m in (hea_member, dat_member):
                (Path(td) / Path(m).name).write_bytes(self.zf.read(m))
            rec = wfdb.rdrecord(str(Path(td) / stem), channels=[idx])
        if rec.sig_name[0] != self.lead:
            raise ValueError(f"{stem}: got {rec.sig_name[0]!r}, expected {self.lead!r}")
        sig = rec.p_signal[:, 0].astype(np.float64)
        if not np.isfinite(sig).all():
            raise ValueError(f"{stem}: non-finite samples in lead {self.lead!r}")
        return sig


def build_metadata(records: pd.DataFrame) -> pd.DataFrame:
    """`records`: one row per record with at least a 'reason' free-text column (parsed from
    header comments) plus 'patient', 'sex', 'age'. Adds dx_group/is_mi/is_hc/sex_male."""
    meta = records.copy()
    meta["dx_group"] = meta["reason"].map(map_group)
    meta["is_mi"] = (meta["dx_group"] == "MI").astype(int)
    meta["is_hc"] = (meta["dx_group"] == "HC").astype(int)
    meta["sex_male"] = meta["sex"].map({"male": 1, "female": 0})
    return meta


def cohort_demographics(meta: pd.DataFrame) -> pd.DataFrame:
    """Per-diagnostic-group subject counts, % male, and age mean/sd — the table to check
    before trusting any probe on this cohort (see docs/methodology.md's standing rule on
    demographic confounds)."""
    subj = meta.drop_duplicates("patient")[["patient", "dx_group", "sex", "age"]]
    return (
        subj[subj["dx_group"].notna()]
        .groupby("dx_group")
        .agg(n_subjects=("patient", "size"),
             pct_male=("sex", lambda s: 100 * (s == "male").mean()),
             age_mean=("age", "mean"),
             age_sd=("age", "std"))
        .round(1)
        .sort_values("n_subjects", ascending=False)
    )


def mi_vs_hc_confound_test(meta: pd.DataFrame) -> dict:
    """Chi-square (sex) and Welch t-test (age) for MI against HC subjects — the exact test
    that found PTB-DB's headline MI-vs-HC AUROC was substantially an age effect."""
    subj = meta.drop_duplicates("patient")[["patient", "dx_group", "sex", "age"]]
    mi_hc = subj[subj["dx_group"].isin(["MI", "HC"])]
    tab = pd.crosstab(mi_hc["dx_group"], mi_hc["sex"])
    chi2, p_sex, _, _ = sps.chi2_contingency(tab)
    a_mi = mi_hc.loc[mi_hc.dx_group == "MI", "age"].dropna()
    a_hc = mi_hc.loc[mi_hc.dx_group == "HC", "age"].dropna()
    t_age, p_age = sps.ttest_ind(a_mi, a_hc, equal_var=False)
    return dict(sex_table=tab, chi2=chi2, p_sex=p_sex,
                age_mi_mean=float(a_mi.mean()), age_hc_mean=float(a_hc.mean()),
                t_age=t_age, p_age=p_age)
