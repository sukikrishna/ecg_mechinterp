#!/usr/bin/env python3
"""RQ1: do different encoders recover a shared representation of the same clinical concept?

Computes cross-encoder similarity (linear/RBF CKA, Procrustes, SVCCA, kNN overlap) at every
configured relative depth, each against a permutation null and a within-encoder noise
ceiling (docs/methodology.md rule 3) — and, if any `_rand` encoder is included, the
random-init floor that number should really be read against (docs/findings.md finding #4).
Also computes concept-conditional similarity: does global similarity hold up when restricted
to the subspace that actually separates a concept's classes?

Usage:
    python3 scripts/run_similarity_sweep.py --dataset ptbxl \\
        --encoders ecgfounder_1lead ecgfounder_1lead_rand clef_m clef_m_rand ecgjepa
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecg_mechinterp.activation.extraction import extract_token_activations, flat_tokens, token_fractions  # noqa: E402
from ecg_mechinterp.config import Config  # noqa: E402
from ecg_mechinterp.data.loader import load_windows  # noqa: E402
from ecg_mechinterp.data.splits import make_patient_splits  # noqa: E402
from ecg_mechinterp.models.registry import build_encoder_registry  # noqa: E402
from ecg_mechinterp.similarity.controls import similarity_with_reference, subsample_rows, within_encoder_ceiling  # noqa: E402
from ecg_mechinterp.similarity.metrics import linear_cka  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ptbxl", "ptbdb"], default="ptbxl")
    parser.add_argument("--encoders", nargs="+", required=True)
    parser.add_argument("--concept", default=None, help="binary concept column in win_meta "
                        "for the concept-conditional similarity check (e.g. myocardial_infarction)")
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
    fractions = token_fractions(cfg.tokens_per_window)

    loaded, failed = build_encoder_registry(cfg.encoders, cfg.ckpt_dir, device=cfg.device, seed=cfg.seed)
    keys = list(loaded)

    acts = {}
    for key, enc in loaded.items():
        x_model = np.stack([enc.ecg_model.preprocess(x_raw[i]).squeeze(0).numpy() for i in range(len(x_raw))])
        layer_map = enc.layers_at_relative_depth(cfg.relative_depths)
        acts[key] = dict(layer_map=layer_map, model_input=x_model,
                         activations=extract_token_activations(
                             enc, x_model, list(set(layer_map.values())), fractions,
                             time_pool=cfg.time_pool, batch_size=cfg.batch_size, device=cfg.device,
                             desc=key))

    ceiling = {}
    for key in keys:
        for depth, layer in acts[key]["layer_map"].items():
            ia, ib = splits["train_a"], splits["train_b"]
            n = min(len(ia), len(ib))
            a = flat_tokens(acts[key]["activations"][layer], ia[:n])
            b = flat_tokens(acts[key]["activations"][layer], ib[:n])
            ceiling[(key, depth)] = within_encoder_ceiling(a, b, n_subsample=4000, seed=cfg.seed)
    print("within-encoder noise ceiling:")
    print(pd.DataFrame(ceiling).T.round(3).to_string())

    te_idx = splits["test"]
    rows = []
    for depth in cfg.relative_depths:
        reps = {key: flat_tokens(acts[key]["activations"][acts[key]["layer_map"][depth]], te_idx) for key in keys}
        patients_tok = np.repeat(win_meta["patient"].values[te_idx], cfg.tokens_per_window)
        for ka, kb in combinations(keys, 2):
            a, b, p = subsample_rows(reps[ka], reps[kb], patients_tok, n=4000, seed=cfg.seed)
            res = similarity_with_reference(a, b, p, n_permutation=cfg.n_permutation, seed=cfg.seed)
            ceil = float(np.mean([ceiling[(ka, depth)]["cka_linear"], ceiling[(kb, depth)]["cka_linear"]]))
            rows.append(dict(depth=depth, a=ka, b=kb, ceiling=round(ceil, 3),
                             cka_over_ceiling=round(res["cka_linear"] / max(1e-6, ceil), 3),
                             **{k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}))
    sim_df = pd.DataFrame(rows)
    sim_path = os.path.join(cfg.out_dir, "tables", "representation_similarity.csv")
    sim_df.to_csv(sim_path, index=False)
    print(f"\nwrote {sim_path}")
    print(sim_df[["depth", "a", "b", "cka_linear", "cka_linear_null", "ceiling", "cka_over_ceiling"]].to_string(index=False))

    real = sim_df[~sim_df.a.str.endswith("_rand") & ~sim_df.b.str.endswith("_rand")]
    rand = sim_df[sim_df.a.str.endswith("_rand") | sim_df.b.str.endswith("_rand")]
    if len(real) and len(rand):
        print(f"\npretrained-pair mean CKA: {real.cka_linear.mean():.3f} | "
              f"random-encoder-pair mean CKA (the floor, see docs/methodology.md rule 3): "
              f"{rand.cka_linear.mean():.3f}")

    if args.concept:
        cc_rows = []
        for depth in cfg.relative_depths:
            mask = win_meta.loc[te_idx, args.concept].notna().values
            idx = te_idx[mask]
            y_tok = np.repeat(win_meta.loc[idx, args.concept].astype(int).values, cfg.tokens_per_window)
            if len(np.unique(y_tok)) < 2:
                continue
            projected = {}
            for key in keys:
                z_tok = flat_tokens(acts[key]["activations"][acts[key]["layer_map"][depth]], idx)
                classes = np.unique(y_tok)
                means = np.stack([z_tok[y_tok == c].mean(0) for c in classes])
                diffs = means - means.mean(0, keepdims=True)
                u, _, _ = np.linalg.svd(diffs.T, full_matrices=False)
                projected[key] = z_tok @ u[:, : min(8, u.shape[1])]
            for ka, kb in combinations(keys, 2):
                a, b = subsample_rows(projected[ka], projected[kb], n=4000, seed=cfg.seed)
                cc_rows.append(dict(depth=depth, concept=args.concept, a=ka, b=kb,
                                    cka=round(linear_cka(a, b), 3)))
        cc_df = pd.DataFrame(cc_rows)
        cc_path = os.path.join(cfg.out_dir, "tables", "concept_conditional_similarity.csv")
        cc_df.to_csv(cc_path, index=False)
        print(f"\nwrote {cc_path}")
        print(cc_df.to_string(index=False))

    if failed:
        print(f"\nencoders that failed to load (skipped): {failed}")


if __name__ == "__main__":
    main()
