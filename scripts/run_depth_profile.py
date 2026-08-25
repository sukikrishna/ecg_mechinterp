#!/usr/bin/env python3
"""Per-layer, per-concept probing across every loaded encoder — the depth-decodability
profile (docs/findings.md finding/discrepancy on depth). Always includes a random-init
control per encoder in `cfg.encoders` if configured (docs/methodology.md rule 3), and reports
a patient-level bootstrap CI for every AUROC/Spearman rho (rule 1).

Usage:
    python3 scripts/run_depth_profile.py --dataset ptbxl --encoders ecgfounder_1lead clef_m ecgjepa
    python3 scripts/run_depth_profile.py --dataset ptbdb --encoders clef_m clef_m_rand
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_mechinterp.activation.extraction import extract_layer_pooled  # noqa: E402
from ecg_mechinterp.config import Config  # noqa: E402
from ecg_mechinterp.data.loader import load_windows  # noqa: E402
from ecg_mechinterp.data.splits import make_patient_splits  # noqa: E402
from ecg_mechinterp.models.registry import build_encoder_registry  # noqa: E402
from ecg_mechinterp.probing.probe import fit_probe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ptbxl", "ptbdb"], default="ptbxl")
    parser.add_argument("--encoders", nargs="+", required=True)
    parser.add_argument("--concepts", nargs="+", default=None, help="concept mask columns in "
                        "win_meta; defaults to {mi_vs_hc} for ptbdb, all PTBXL concept columns "
                        "for ptbxl")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ckpt-dir", default="weights")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    cfg = Config(dataset=args.dataset, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir,
                out_dir=args.out_dir, max_records=args.max_records, encoders=tuple(args.encoders))
    cfg.ensure_dirs()

    x_raw, win_meta = load_windows(cfg)
    win_meta["dx_group"] = win_meta.get("dx_group", pd.Series("NA", index=win_meta.index)).fillna("NA")
    win_meta["sex"] = win_meta["sex"].fillna("NA")
    splits = make_patient_splits(win_meta, stratify_cols=("dx_group", "sex"), seed=cfg.seed)

    loaded, failed = build_encoder_registry(cfg.encoders, cfg.ckpt_dir, device=cfg.device, seed=cfg.seed)

    if args.concepts:
        concepts = args.concepts
    elif cfg.dataset == "ptbdb":
        concepts = ["is_mi"]
    else:
        concepts = [c for c in win_meta.columns if c in (
            "atrial_fibrillation", "bundle_branch_block", "normal_rhythm",
            "left_ventricular_hypertrophy", "myocardial_infarction")]

    rows = []
    for key, enc in loaded.items():
        x_model = np.stack([enc.ecg_model.preprocess(x_raw[i]).squeeze(0).numpy() for i in range(len(x_raw))])
        for layer in enc.layer_names():
            z_train = extract_layer_pooled(enc, x_model[splits["train"]], layer,
                                           batch_size=cfg.batch_size, device=cfg.device)
            z_test = extract_layer_pooled(enc, x_model[splits["test"]], layer,
                                          batch_size=cfg.batch_size, device=cfg.device)
            for concept in concepts:
                y_train = win_meta.loc[splits["train"], concept].astype(int).values
                y_test = win_meta.loc[splits["test"], concept].astype(int).values
                patients_test = win_meta.loc[splits["test"], "patient"].values
                if len(np.unique(y_train)) < 2:
                    continue
                res = fit_probe(z_train, y_train, z_test, y_test, patients_test,
                                kind="binary", n_bootstrap=cfg.n_bootstrap, seed=cfg.seed)
                rows.append(dict(encoder=key, layer=layer, concept=concept,
                                 auroc=round(res["value"], 3), lo=round(res["lo"], 3), hi=round(res["hi"], 3),
                                 n_test=len(y_test)))
                print(f"[{key}] {layer} {concept}: AUROC {res['value']:.3f} "
                      f"[{res['lo']:.3f}, {res['hi']:.3f}] n={len(y_test)}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(cfg.out_dir, "tables", "depth_profile.csv")
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    if failed:
        print(f"encoders that failed to load (skipped): {failed}")


if __name__ == "__main__":
    main()
