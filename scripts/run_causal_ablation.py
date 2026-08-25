#!/usr/bin/env python3
"""RQ4: does cross-seed feature stability predict causal importance?

Shortlists SAE features by gradient attribution to a probe's score, verifies the shortlist by
actually ablating each one and re-measuring the probe (docs/methodology.md rule 4: attribution
alone is not the causal claim, ablation is), then correlates each feature's cross-seed
stability (from run_stability_suite.py's dictionary matching) against its ablation effect,
reporting both the raw and density-partialled Spearman rho (density is a plausible common
cause of both quantities).

Usage:
    python3 scripts/run_causal_ablation.py --dataset ptbdb --encoder clef_m \\
        --sae-store results/sae_store.pt --concept is_mi
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

from ecg_mechinterp.bias.demographic import partial_spearman  # noqa: E402
from ecg_mechinterp.causal.ablation import ablation_effect  # noqa: E402
from ecg_mechinterp.causal.attribution import TorchProbe, feature_attribution  # noqa: E402
from ecg_mechinterp.config import Config  # noqa: E402
from ecg_mechinterp.data.loader import load_windows  # noqa: E402
from ecg_mechinterp.data.splits import make_patient_splits  # noqa: E402
from ecg_mechinterp.models.registry import build_encoder_registry  # noqa: E402
from ecg_mechinterp.probing.probe import fit_probe  # noqa: E402
from ecg_mechinterp.sae.stability import dictionary_matching  # noqa: E402
from scipy import stats as sps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ptbxl", "ptbdb"], default="ptbdb")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--depth", type=float, default=0.75)
    parser.add_argument("--sae-store", default="results/sae_store.pt")
    parser.add_argument("--concept", required=True)
    parser.add_argument("--top-features", type=int, default=24)
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

    idx_train = splits["train"][win_meta.loc[splits["train"], args.concept].notna().values]
    idx_test = splits["test"][win_meta.loc[splits["test"], args.concept].notna().values]
    y_train = win_meta.loc[idx_train, args.concept].astype(int).values
    y_test = win_meta.loc[idx_test, args.concept].astype(int).values
    patients_test = win_meta.loc[idx_test, "patient"].values

    from ecg_mechinterp.activation.extraction import extract_pooled_embedding
    emb_train = extract_pooled_embedding(enc, x_model[idx_train], batch_size=cfg.batch_size, device=cfg.device)
    emb_test = extract_pooled_embedding(enc, x_model[idx_test], batch_size=cfg.batch_size, device=cfg.device)
    probe_fit = fit_probe(emb_train, y_train, emb_test, y_test, patients_test, kind="binary", seed=cfg.seed)
    probe = TorchProbe(probe_fit["model"]).to(cfg.device)
    print(f"probe on [{args.encoder}] {args.concept}: AUROC {probe_fit['value']:.3f}")

    rng = np.random.default_rng(cfg.seed)
    eval_idx = idx_test if len(idx_test) <= args.eval_windows else np.sort(
        rng.choice(idx_test, args.eval_windows, replace=False))
    eval_x, eval_y = x_model[eval_idx], win_meta.loc[eval_idx, args.concept].astype(int).values

    attr, density = feature_attribution(enc, probe, sae, layer, eval_x, time_pool=cfg.time_pool, device=cfg.device)
    order = np.argsort(-np.abs(attr))
    shortlist = [int(f) for f in order[: args.top_features] if density[f] > 1e-4]

    causal_rows = []
    for f in shortlist:
        eff = ablation_effect(enc, probe, sae, layer, [f], eval_x, eval_y, time_pool=cfg.time_pool, device=cfg.device)
        causal_rows.append(dict(feature=f, attribution=float(attr[f]), density=float(density[f]), **eff))
    causal_df = pd.DataFrame(causal_rows).sort_values("delta_score", ascending=False)
    causal_path = os.path.join(cfg.out_dir, "tables", "causal_effects.csv")
    causal_df.to_csv(causal_path, index=False)
    print(f"\nwrote {causal_path}")
    print(causal_df.head(12).round(4).to_string(index=False))

    other_tags = [t for t in tags if t != tags[0]]
    if other_tags and len(causal_df) >= 8:
        stab_vec = np.array([dictionary_matching(sae, sae_store[t])["per_feature"] for t in other_tags]).mean(0)
        f = causal_df.feature.values
        s = stab_vec[f]
        eff = causal_df.delta_score.values
        dens = causal_df.density.values
        rho, p = sps.spearmanr(s, eff)
        rho_partial = partial_spearman(s, eff, dens)
        print(f"\nRQ4: stability vs. ablation effect — rho={rho:.3f} p={p:.4f}, "
              f"density-partialled rho={rho_partial:.3f} (docs/methodology.md rule 4: "
              "read the partialled value, not the raw one, as the real answer)")
    else:
        print("\nRQ4 skipped: need >=2 SAE seeds in --sae-store and >=8 shortlisted features")

    if failed:
        print(f"\nencoders that failed to load (skipped): {failed}")


if __name__ == "__main__":
    main()
