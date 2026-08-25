#!/usr/bin/env python3
"""Train one SAE per (encoder, relative depth, method, sparsity, seed) grid cell.

Before trusting any of this, run the convergence check (docs/methodology.md rule 2): this
script runs it once, on the first encoder/depth combination, and prints the FVU-vs-MMCS
divergence table so the chosen `--steps` can be checked against it rather than assumed.

Usage:
    python3 scripts/train_sae_sweep.py --dataset ptbdb --encoders clef_m --seeds 0 1 2 3 4 \\
        --steps 34920
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

from ecg_mechinterp.activation.extraction import extract_token_activations, flat_tokens, token_fractions  # noqa: E402
from ecg_mechinterp.config import Config  # noqa: E402
from ecg_mechinterp.data.loader import load_windows  # noqa: E402
from ecg_mechinterp.data.splits import make_patient_splits  # noqa: E402
from ecg_mechinterp.models.registry import build_encoder_registry  # noqa: E402
from ecg_mechinterp.sae.train import sae_report, track_convergence, train_sae  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["ptbxl", "ptbdb"], default="ptbdb")
    parser.add_argument("--encoders", nargs="+", required=True)
    parser.add_argument("--depth", type=float, default=0.75, help="relative depth to train the SAE at")
    parser.add_argument("--methods", nargs="+", default=["topk"])
    parser.add_argument("--k", nargs="+", type=int, default=[32])
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=34_920)
    parser.add_argument("--skip-convergence-check", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--ckpt-dir", default="weights")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    cfg = Config(dataset=args.dataset, data_dir=args.data_dir, ckpt_dir=args.ckpt_dir,
                out_dir=args.out_dir, max_records=args.max_records, encoders=tuple(args.encoders),
                sae_max_hidden=8192)
    cfg.ensure_dirs()

    x_raw, win_meta = load_windows(cfg)
    win_meta["dx_group"] = win_meta.get("dx_group", pd.Series("NA", index=win_meta.index)).fillna("NA")
    win_meta["sex"] = win_meta["sex"].fillna("NA")
    splits = make_patient_splits(win_meta, stratify_cols=("dx_group", "sex"), seed=cfg.seed)
    fractions = token_fractions(cfg.tokens_per_window)

    loaded, failed = build_encoder_registry(cfg.encoders, cfg.ckpt_dir, device=cfg.device, seed=cfg.seed)

    sae_store = {}
    rows = []
    checked_convergence = False
    for key, enc in loaded.items():
        x_model = np.stack([enc.ecg_model.preprocess(x_raw[i]).squeeze(0).numpy() for i in range(len(x_raw))])
        layer = enc.layers_at_relative_depth([args.depth])[args.depth]
        acts = extract_token_activations(enc, x_model, [layer], fractions, time_pool=cfg.time_pool,
                                         batch_size=cfg.batch_size, device=cfg.device, desc=key)
        a_train = flat_tokens(acts[layer], splits["train"])
        a_test = flat_tokens(acts[layer], splits["test"])
        d_in = a_train.shape[1]

        if not args.skip_convergence_check and not checked_convergence:
            print(f"\nconvergence check on [{key}] {layer} (docs/methodology.md rule 2) — "
                  "confirm MMCS is not still rising sharply at --steps before trusting the sweep below:")
            conv = track_convergence(a_train, a_test, d_hidden=min(args.expansion * d_in, cfg.sae_max_hidden),
                                     budgets=tuple(sorted({args.steps // 30, args.steps // 10, args.steps // 3, args.steps})),
                                     seeds=(0, 1), device=cfg.device)
            print(conv.to_string(index=False))
            checked_convergence = True

        for method in args.methods:
            ks = args.k if method == "topk" else [args.k[0]]
            for k in ks:
                for seed in args.seeds:
                    d_hidden = min(args.expansion * d_in, cfg.sae_max_hidden)
                    sae = train_sae(a_train, d_hidden=d_hidden, method=method, k=k, seed=seed,
                                    steps=args.steps, batch=cfg.sae_batch, lr=cfg.sae_lr, device=cfg.device)
                    sae_store[(key, layer, method, k, args.expansion, seed)] = sae
                    rep = sae_report(sae, a_test, device=cfg.device)
                    rows.append(dict(encoder=key, layer=layer, method=method, k=k,
                                     expansion=args.expansion, seed=seed, d_in=d_in,
                                     **{a: round(b, 4) for a, b in rep.items()}))
                    print(f"[{key}] {layer} {method} k={k} exp={args.expansion} seed={seed}: "
                          f"fvu={rep['fvu']:.4f} dead_frac={rep['dead_frac']:.4f}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(cfg.out_dir, "tables", "sae_quality.csv")
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    sae_path = os.path.join(cfg.out_dir, "sae_store.pt")
    torch.save({str(k): v for k, v in sae_store.items()}, sae_path)
    print(f"wrote {sae_path} — used by scripts/run_stability_suite.py and run_causal_ablation.py")
    if failed:
        print(f"encoders that failed to load (skipped): {failed}")


if __name__ == "__main__":
    main()
