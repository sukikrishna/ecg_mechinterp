#!/usr/bin/env python3
"""The demographic-pathway test (docs/findings.md finding #1, the flagship result this repo
consolidates): ablate the SAE features most attributed to a *protected* concept (e.g. sex),
then re-measure a *target* clinical concept (e.g. MI). A drop larger than a density-matched
control's is evidence the target concept's decoding partly runs through the protected
pathway — not proof by itself that the model is "biased" in a normative sense, but direct
evidence about what the representation actually encodes.

Run this on PTB-XL for myocardial_infarction and left_ventricular_hypertrophy (both age-skewed
diagnoses never checked this way before — docs/findings.md discrepancy #2) as well as on
PTB-DB, where it was first established.

Usage:
    python3 scripts/run_demographic_bias_check.py --dataset ptbdb --encoder clef_m \\
        --sae-store results/sae_store.pt --protected sex --target is_mi
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_mechinterp.activation.extraction import extract_pooled_embedding  # noqa: E402
from ecg_mechinterp.bias.demographic import demographic_pathway_test  # noqa: E402
from ecg_mechinterp.causal.attribution import TorchProbe, feature_attribution  # noqa: E402
from ecg_mechinterp.config import Config  # noqa: E402
from ecg_mechinterp.data.loader import load_windows  # noqa: E402
from ecg_mechinterp.data.splits import make_patient_splits  # noqa: E402
from ecg_mechinterp.models.registry import build_encoder_registry  # noqa: E402
from ecg_mechinterp.probing.probe import fit_probe  # noqa: E402


def _fit(enc, cfg, x_model, win_meta, splits, concept):
    idx_train = splits["train"][win_meta.loc[splits["train"], concept].notna().values]
    idx_test = splits["test"][win_meta.loc[splits["test"], concept].notna().values]
    y_train = win_meta.loc[idx_train, concept].astype(int).values
    y_test = win_meta.loc[idx_test, concept].astype(int).values
    emb_train = extract_pooled_embedding(enc, x_model[idx_train], batch_size=cfg.batch_size, device=cfg.device)
    emb_test = extract_pooled_embedding(enc, x_model[idx_test], batch_size=cfg.batch_size, device=cfg.device)
    fit = fit_probe(emb_train, y_train, emb_test, y_test, win_meta.loc[idx_test, "patient"].values,
                    kind="binary", seed=cfg.seed)
    return fit, idx_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ptbxl", "ptbdb"], default="ptbdb")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--depth", type=float, default=0.75)
    parser.add_argument("--sae-store", default="results/sae_store.pt")
    parser.add_argument("--protected", default="sex_male", help="binary concept column to "
                        "ablate for, e.g. sex_male")
    parser.add_argument("--target", nargs="+", required=True, help="one or more binary "
                        "concept columns to measure after ablation, e.g. is_mi or "
                        "myocardial_infarction left_ventricular_hypertrophy")
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--eval-windows", type=int, default=384)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ckpt-dir", default="weights")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    cfg = Config(dataset=args.dataset, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir,
                out_dir=args.out_dir, max_records=args.max_records, encoders=(args.encoder,))
    cfg.ensure_dirs()

    sae_store = torch.load(args.sae_store, weights_only=False)

    x_raw, win_meta = load_windows(cfg)
    win_meta["dx_group"] = win_meta.get("dx_group", pd.Series("NA", index=win_meta.index)).fillna("NA")
    win_meta["sex"] = win_meta["sex"].fillna("NA")
    splits = make_patient_splits(win_meta, stratify_cols=("dx_group", "sex"), seed=cfg.seed)

    loaded, failed = build_encoder_registry(cfg.encoders, cfg.ckpt_dir, device=cfg.device, seed=cfg.seed)
    enc = loaded[args.encoder]
    x_model = np.stack([enc.ecg_model.preprocess(x_raw[i]).squeeze(0).numpy() for i in range(len(x_raw))])
    layer = enc.layers_at_relative_depth([args.depth])[args.depth]

    tags = [t for t in sae_store if t.startswith(f"('{args.encoder}', '{layer}'")]
    if not tags:
        raise SystemExit(f"no SAE found in {args.sae_store} for encoder={args.encoder} layer={layer}")
    sae = sae_store[tags[0]]

    concepts = [args.protected, *args.target]
    probes, eval_windows, eval_labels = {}, {}, {}
    for concept in concepts:
        fit, idx_test, y_test = _fit(enc, cfg, x_model, win_meta, splits, concept)
        print(f"probe on [{args.encoder}] {concept}: AUROC {fit['value']:.3f}")
        probes[concept] = TorchProbe(fit["model"]).to(cfg.device)
        rng = np.random.default_rng(cfg.seed)
        sel = idx_test if len(idx_test) <= args.eval_windows else np.sort(
            rng.choice(idx_test, args.eval_windows, replace=False))
        eval_windows[concept] = x_model[sel]
        eval_labels[concept] = win_meta.loc[sel, concept].astype(int).values

    protected_x = eval_windows[args.protected]
    attr, density = feature_attribution(enc, probes[args.protected], sae, layer, protected_x,
                                        time_pool=cfg.time_pool, device=cfg.device)
    top_features = [int(f) for f in np.argsort(-np.abs(attr))[: args.top_features] if density[f] > 1e-4]

    bias_df = demographic_pathway_test(
        enc, sae, layer, protected_density=density, top_features=top_features,
        probes=probes, eval_windows=eval_windows, eval_labels=eval_labels, subgroup=None,
        time_pool=cfg.time_pool, device=cfg.device, seed=cfg.seed,
    )
    bias_path = os.path.join(cfg.out_dir, "tables", "demographic_pathway.csv")
    bias_df.to_csv(bias_path, index=False)
    print(f"\nwrote {bias_path}")
    print(bias_df.to_string(index=False))
    print(
        "\nRead this as: the '{p}' row under measured_on='{p}' confirms the intervention worked; "
        "each target row's auroc_drop, compared against its density_matched_control row, is the "
        "actual demographic-pathway evidence (docs/methodology.md rule 4).".format(p=args.protected)
    )

    if failed:
        print(f"\nencoders that failed to load (skipped): {failed}")


if __name__ == "__main__":
    main()
