#!/usr/bin/env python3
"""Verify that a local PTB-XL or PTB-DB pull matches the exact corpus the three source
analyses this repo consolidates were run on (see docs/data.md for the reference numbers and
why they matter).

The three source analyses each pulled data independently, at different times, on different
machines. Before trusting a new result as comparable to a prior one — which is the entire
point of consolidating them into one repo — this checks that "PTB-XL" and "PTB-DB" mean the
same bytes every time, not just the same dataset name. Run this right after
scripts/download_ptbxl.sh / download_ptbdb.sh, and again on any new machine (a labmate's
laptop, Kaggle, etc.) before comparing its numbers to anyone else's.

Usage:
    python3 scripts/verify_data.py ptbxl [--root data/raw/ptb-xl]
    python3 scripts/verify_data.py ptbdb [--root data/raw/ptbdb]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_mechinterp.data.ptbdb import build_metadata  # noqa: E402

# Reference numbers, taken directly from what each source analysis reported for its own pull
# — see docs/data.md for the full explanation and citations.
PTBXL_EXPECTED_VERSION = "1.0.3"
PTBXL_EXPECTED_RECORDS = 21_799  # docs/preliminary-results.md: "2 fewer than the commonly
# cited 21,801 -- unexplained, negligible". If your pull gives exactly 21,801, PhysioNet's
# distribution changed since that run; note it in docs/data.md rather than silently using it.

PTBDB_EXPECTED_VERSION = "1.0.0"
PTBDB_EXPECTED_RECORDS = 549
PTBDB_EXPECTED_SUBJECTS = 290
PTBDB_EXPECTED_GROUPS = {"MI": 148, "HC": 52}  # subject-level, per the CLEF SAE convergence
# study's stage-1 gate ("148 MI / 52 HC / 22 no-summary, exactly") and the RQ1-4 notebook's
# independent count (549 records / 290 subjects) -- both analyses' numbers agree, which is
# itself evidence the two pulls were the same corpus.
PTBDB_EXPECTED_NO_SUMMARY = 22


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_ptbxl(root: Path) -> bool:
    ok = True
    db_csv = root / "ptbxl_database.csv"
    if not db_csv.exists():
        print(f"FAIL: {db_csv} not found — run scripts/download_ptbxl.sh first")
        return False

    meta = pd.read_csv(db_csv, index_col="ecg_id")
    n_records = len(meta)
    print(f"PTB-XL records: {n_records} (expected {PTBXL_EXPECTED_RECORDS})")
    if n_records != PTBXL_EXPECTED_RECORDS:
        print(f"  MISMATCH — see docs/data.md before comparing results against prior runs")
        ok = False

    sums_file = root / "SHA256SUMS.txt"
    if sums_file.exists():
        mismatches = []
        for line in sums_file.read_text().splitlines():
            digest, _, name = line.partition(" ")
            name = name.lstrip("*").strip()
            fpath = root / name
            if not fpath.exists() or fpath.is_dir():
                continue  # SHA256SUMS.txt lists every record file; checking the top-level
                # metadata files is enough evidence of a correct pull without re-hashing all
                # ~43k waveform files on every run.
            if name not in ("ptbxl_database.csv", "scp_statements.csv", "LICENSE.txt"):
                continue
            actual = _sha256(fpath)
            if actual != digest:
                mismatches.append(name)
        if mismatches:
            print(f"FAIL: checksum mismatch on {mismatches}")
            ok = False
        else:
            print("OK: ptbxl_database.csv / scp_statements.csv / LICENSE.txt checksums match "
                  "PhysioNet's own SHA256SUMS.txt")
    else:
        print("NOTE: no SHA256SUMS.txt found in the pull — can't verify checksums, only counts")

    print(f"Expected PTB-XL version: {PTBXL_EXPECTED_VERSION} (the download script pins this "
          f"via a version-specific URL, so this is a sanity check, not the real guard)")
    return ok


def verify_ptbdb(root: Path) -> bool:
    import wfdb

    ok = True
    rows = []
    for hea in sorted(root.rglob("*.hea")):
        rec_id = str(hea.relative_to(root)).removesuffix(".hea")
        header = wfdb.rdheader(str(hea)[: -len(".hea")])
        comments = {}
        for line in header.comments:
            line = line.strip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            comments[key.strip().lower()] = val.strip()
        patient = Path(rec_id).parent.name
        rows.append(dict(record=rec_id, patient=patient,
                          reason=comments.get("reason for admission", "n/a"),
                          sex=comments.get("sex"), age=comments.get("age")))
    if not rows:
        print(f"FAIL: no .hea files found under {root} — run scripts/download_ptbdb.sh first")
        return False

    records = pd.DataFrame(rows)
    meta = build_metadata(records)
    n_records, n_subjects = len(meta), meta["patient"].nunique()
    print(f"PTB-DB records: {n_records} (expected {PTBDB_EXPECTED_RECORDS})")
    print(f"PTB-DB subjects: {n_subjects} (expected {PTBDB_EXPECTED_SUBJECTS})")
    if n_records != PTBDB_EXPECTED_RECORDS or n_subjects != PTBDB_EXPECTED_SUBJECTS:
        print("  MISMATCH — see docs/data.md before comparing results against prior runs")
        ok = False

    subj_groups = meta.drop_duplicates("patient")["dx_group"].value_counts(dropna=False)
    print("Subject-level diagnostic groups:")
    print(subj_groups.to_string())
    n_no_summary = int(subj_groups.get(float("nan"), 0)) if subj_groups.index.hasnans else 0
    for group, expected_n in PTBDB_EXPECTED_GROUPS.items():
        actual_n = int(subj_groups.get(group, 0))
        if actual_n != expected_n:
            print(f"  MISMATCH on group {group}: got {actual_n}, expected {expected_n}")
            ok = False
    if n_no_summary != PTBDB_EXPECTED_NO_SUMMARY:
        print(f"  NOTE: {n_no_summary} no-summary subjects, expected {PTBDB_EXPECTED_NO_SUMMARY}")

    print(f"Expected PTB-DB version: {PTBDB_EXPECTED_VERSION} (the download script pins this "
          f"via a version-specific URL, so this is a sanity check, not the real guard)")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["ptbxl", "ptbdb"])
    parser.add_argument("--root", default=None)
    args = parser.parse_args()

    if args.dataset == "ptbxl":
        ok = verify_ptbxl(Path(args.root or "data/raw/ptb-xl"))
    else:
        ok = verify_ptbdb(Path(args.root or "data/raw/ptbdb"))

    print("\nPASS" if ok else "\nFAIL — do not treat this pull as consistent with prior runs "
                              "until the mismatch above is understood (see docs/data.md)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
