"""PTB-XL loading utilities: metadata, waveforms, and concept labels.

Ported unchanged from the ecg_interp repo (src/ecg_interp/data/ptbxl.py). This is the
large-N dataset the consolidated findings (docs/findings.md) recommend moving all analyses
to, once statistical power and demographic-confound checks matter — see docs/methodology.md.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import wfdb

# Concept -> SCP-ECG code membership. `codes` are literal SCP-ECG codes (as used in
# scp_statements.csv); `diagnostic_class` selects every code under that PTB-XL superclass.
# Run `PTBXL.check_concept_codes()` after downloading to confirm these against the real file —
# they're the standard PTB-XL codes but haven't been cross-checked against a local copy yet.
CONCEPT_DEFINITIONS = {
    "atrial_fibrillation": {"codes": ["AFIB"]},
    "bundle_branch_block": {"codes": ["CLBBB", "CRBBB", "ILBBB", "IRBBB"]},
    "normal_rhythm": {"diagnostic_class": "NORM"},
    "left_ventricular_hypertrophy": {"codes": ["LVH"]},
    "myocardial_infarction": {"diagnostic_class": "MI"},
}


@dataclass
class PTBXL:
    root: Path
    metadata: pd.DataFrame
    scp_statements: pd.DataFrame

    @classmethod
    def load(cls, root: Union[str, Path]) -> "PTBXL":
        root = Path(root)
        metadata = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
        metadata["scp_codes"] = metadata["scp_codes"].apply(ast.literal_eval)
        scp_statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
        return cls(root=root, metadata=metadata, scp_statements=scp_statements)

    def check_concept_codes(self) -> dict:
        """Report which CONCEPT_DEFINITIONS codes are missing from scp_statements.csv — a
        sign a code name needs fixing before trusting concept_labels()."""
        missing = {}
        for concept, spec in CONCEPT_DEFINITIONS.items():
            codes = spec.get("codes", [])
            not_found = [c for c in codes if c not in self.scp_statements.index]
            if not_found:
                missing[concept] = not_found
        return missing

    def concept_labels(self) -> pd.DataFrame:
        """Boolean DataFrame (ecg_id x concept): True if the record carries any SCP code
        belonging to that concept."""
        labels = pd.DataFrame(index=self.metadata.index)
        for concept, spec in CONCEPT_DEFINITIONS.items():
            if "diagnostic_class" in spec:
                concept_codes = set(
                    self.scp_statements.index[
                        self.scp_statements["diagnostic_class"] == spec["diagnostic_class"]
                    ]
                )
            else:
                concept_codes = set(spec["codes"])
            labels[concept] = self.metadata["scp_codes"].apply(
                lambda d: any(code in concept_codes for code in d)
            )
        return labels

    def demographics(self) -> pd.DataFrame:
        """age/sex per ecg_id, for the same confound checks used on PTB-DB (see
        docs/methodology.md) — PTB-XL's own database.csv already carries these columns."""
        return self.metadata[["age", "sex"]].copy()

    def load_waveforms(self, ecg_ids, sampling_rate: int = 100) -> np.ndarray:
        """Load raw waveforms for the given ecg_ids. sampling_rate is 100 or 500 (Hz) — PTB-XL
        ships both as filename_lr (100Hz) and filename_hr (500Hz)."""
        col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        signals = []
        for ecg_id in ecg_ids:
            record_path = self.root / self.metadata.loc[ecg_id, col]
            signal, _ = wfdb.rdsamp(str(record_path))
            signals.append(signal)
        return np.stack(signals)  # (n, samples, 12 leads)


if __name__ == "__main__":
    import sys

    ptbxl = PTBXL.load(sys.argv[1] if len(sys.argv) > 1 else "data/raw/ptb-xl")
    missing = ptbxl.check_concept_codes()
    if missing:
        print("WARNING - codes not found in scp_statements.csv:", missing)
    else:
        print("All concept codes resolved OK.")
    print(ptbxl.concept_labels().sum())
