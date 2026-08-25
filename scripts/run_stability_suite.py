#!/usr/bin/env python3
"""RQ2 (individual SAE feature stability) and RQ3 (subspace stability, stable rank, probe
direction stability), reading the SAE dictionary trained by train_sae_sweep.py.

Every metric here is reported against a null (docs/methodology.md rule 3): dictionary
matching against `random_dictionary_null`, subspace overlap against a shuffled-token null at
the same rank, activation matching implicitly against its own alive-feature intersection.

Usage:
    python3 scripts/run_stability_suite.py --dataset ptbdb --encoders clef_m \\
        --sae-store results/sae_store.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_mechinterp.activation.extraction import extract_token_activations, flat_tokens, token_fractions  # noqa: E402
from ecg_mechinterp.config import Config  # noqa: E402
from ecg_mechinterp.data.loader import load_windows  # noqa: E402
from ecg_mechinterp.data.splits import make_patient_splits  # noqa: E402
from ecg_mechinterp.models.registry import build_encoder_registry  # noqa: E402
from ecg_mechinterp.probing.probe import dims_needed, probe_direction_stability  # noqa: E402
from ecg_mechinterp.sae.stability import activation_matching, dictionary_matching, random_dictionary_null  # noqa: E402
from ecg_mechinterp.sae.train import sae_encode_all  # noqa: E402
from ecg_mechinterp.similarity.subspace import stable_rank_table, subspace_overlap, subspace_scores  # noqa: E402

RANKS = [1, 2, 4, 8, 16, 32, 64]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ptbxl", "ptbdb"], default="ptbdb")
    parser.add_argument("--encoders", nargs="+", required=True)
    parser.add_argument("--depth", type=float, default=0.75)
    parser.add_argument("--sae-store", default="results/sae_store.pt")
    parser.add_argument("--concept", default=None, help="binary concept for probe-direction "
                        "stability + dims_needed, e.g. myocardial_infarction / is_mi")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ckpt-dir", default="weights")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    cfg = Config(dataset=args.dataset, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir,
                out_dir=args.out_dir, max_records=args.max_records, encoders=tuple(args.encoders))
    cfg.ensure_dirs()

    sae_store = torch.load(args.sae_store, weights_only=False)

    x_raw, win_meta = load_windows(cfg)
    win_meta["dx_group"] = win_meta.get("dx_group", pd.Series("NA", index=win_meta.index)).fillna("NA")
    win_meta["sex"] = win_meta["sex"].fillna("NA")
    splits = make_patient_splits(win_meta, stratify_cols=("dx_group", "sex"), seed=cfg.seed)
    fractions = token_fractions(cfg.tokens_per_window)

    loaded, failed = build_encoder_registry(cfg.encoders, cfg.ckpt_dir, device=cfg.device, seed=cfg.seed)
    keys = list(loaded)

    acts, layer_of = {}, {}
    for key, enc in loaded.items():
        x_model = np.stack([enc.ecg_model.preprocess(x_raw[i]).squeeze(0).numpy() for i in range(len(x_raw))])
        layer_of[key] = enc.layers_at_relative_depth([args.depth])[args.depth]
        acts[key] = extract_token_activations(enc, x_model, [layer_of[key]], fractions,
                                              time_pool=cfg.time_pool, batch_size=cfg.batch_size,
                                              device=cfg.device, desc=key)

    # --- RQ2: cross-seed dictionary + activation matching, per encoder -----------------
    stab_rows = []
    for key in keys:
        tags = [t for t in sae_store if t.startswith(f"('{key}', '{layer_of[key]}'")]
        saes = {t: sae_store[t] for t in tags}
        if len(saes) < 2:
            continue
        a_test = flat_tokens(acts[key][layer_of[key]], splits["test"])
        z_by_tag = {t: sae_encode_all(sae, a_test, device=cfg.device) for t, sae in saes.items()}
        dict_vals, act_vals = [], []
        for ta, tb in combinations(tags, 2):
            dict_vals.append(dictionary_matching(saes[ta], saes[tb])["mmcs"])
            act_vals.append(activation_matching(z_by_tag[ta], z_by_tag[tb], seed=cfg.seed)["mmcs"])
        first_sae = next(iter(saes.values()))
        null = random_dictionary_null(first_sae.d_in, first_sae.d_hidden, seed=cfg.seed)
        stab_rows.append(dict(encoder=key, layer=layer_of[key],
                              dict_mmcs=round(float(np.mean(dict_vals)), 3),
                              dict_mmcs_null=round(null, 3),
                              dict_mmcs_adj=round(float(np.mean(dict_vals)) - null, 3),
                              act_mmcs=round(float(np.mean(act_vals)), 3), n_pairs=len(dict_vals)))
    stab_df = pd.DataFrame(stab_rows)
    stab_path = os.path.join(cfg.out_dir, "tables", "feature_stability_cross_seed.csv")
    stab_df.to_csv(stab_path, index=False)
    print(f"wrote {stab_path}")
    print(stab_df.to_string(index=False))

    # --- RQ3: subspace overlap / stable rank, within- and cross-encoder ----------------
    sub_rows = []
    for key in keys:
        ia, ib = splits["train_a"], splits["train_b"]
        n = min(len(ia), len(ib))
        a = flat_tokens(acts[key][layer_of[key]], ia[:n])
        b = flat_tokens(acts[key][layer_of[key]], ib[:n])
        for k in RANKS:
            if k >= min(a.shape[1], b.shape[1]):
                continue
            sub_rows.append(dict(kind="within_encoder_data_split", a=key, b=key, rank=k,
                                 overlap=round(subspace_overlap(subspace_scores(a, k), subspace_scores(b, k)), 3)))
    reps_te = {key: flat_tokens(acts[key][layer_of[key]], splits["test"]) for key in keys}
    for ka, kb in combinations(keys, 2):
        a, b = reps_te[ka], reps_te[kb]
        for k in RANKS:
            if k >= min(a.shape[1], b.shape[1]):
                continue
            sub_rows.append(dict(kind="cross_encoder", a=ka, b=kb, rank=k,
                                 overlap=round(subspace_overlap(subspace_scores(a, k), subspace_scores(b, k)), 3)))
    subspace_df = pd.DataFrame(sub_rows)
    subspace_path = os.path.join(cfg.out_dir, "tables", "subspace_overlap.csv")
    subspace_df.to_csv(subspace_path, index=False)
    stable_rank_df = stable_rank_table(sub_rows, threshold=cfg.stable_rank_threshold)
    print(f"\nwrote {subspace_path}")
    print(f"stable rank at overlap >= {cfg.stable_rank_threshold}:")
    print(stable_rank_df.to_string(index=False))

    # --- RQ3 continued: probe direction stability + dims_needed ------------------------
    if args.concept:
        dir_rows = []
        for key in keys:
            emb = acts[key][layer_of[key]].astype(np.float32).mean(axis=1)  # whole-window pool
            idx_train = splits["train"][win_meta.loc[splits["train"], args.concept].notna().values]
            idx_test = splits["test"][win_meta.loc[splits["test"], args.concept].notna().values]
            y_train = win_meta.loc[idx_train, args.concept].astype(int).values
            y_test = win_meta.loc[idx_test, args.concept].astype(int).values
            if len(np.unique(y_train)) < 2:
                continue
            pds = probe_direction_stability(emb[idx_train], y_train, win_meta.loc[idx_train, "patient"].values,
                                            kind="binary", seed=cfg.seed)
            dn = dims_needed(emb[idx_train], y_train, emb[idx_test], y_test, kind="binary")
            dir_rows.append(dict(encoder=key, concept=args.concept, direction_stability=round(pds, 3),
                                 **{f"k={k}": v for k, v in dn.items()}))
        dir_df = pd.DataFrame(dir_rows)
        dir_path = os.path.join(cfg.out_dir, "tables", "concept_direction_stability.csv")
        dir_df.to_csv(dir_path, index=False)
        print(f"\nwrote {dir_path}")
        print(dir_df.to_string(index=False))

    if failed:
        print(f"\nencoders that failed to load (skipped): {failed}")


if __name__ == "__main__":
    main()
