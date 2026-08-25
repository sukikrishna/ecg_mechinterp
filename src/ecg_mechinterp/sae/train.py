"""SAE training, with the training-budget/convergence check as a first-class, reusable step
rather than something a script re-derives from scratch.

The single most important methodological finding behind this module (see docs/findings.md):
reconstruction quality (FVU) saturates long before cross-seed feature *identity* does, so
`FVU < some threshold` is not a valid stopping criterion for a stability study — it was
observed to keep improving while a stability metric (MMCS) was still moving 8x faster, in the
same training range. `track_convergence` runs the exact sweep that finding came from (train
several budgets, several seeds, report FVU and MMCS together) so a new encoder/layer/dataset
gets checked the same way instead of trusting a borrowed epoch count.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ecg_mechinterp.sae.model import SAE, auxk_loss, fvu, geometric_median
from ecg_mechinterp.sae.stability import dictionary_matching
from ecg_mechinterp.utils import set_seed


def train_sae(activations_train: np.ndarray, d_hidden: int, method: str = "topk", k: int = 32,
              seed: int = 0, steps: int = 4000, batch: int = 4096, lr: float = 3e-4,
              l1_coef: float = 4e-3, use_geometric_median_init: bool = True,
              use_auxk: bool = False, k_aux: Optional[int] = None, resample_every: int = 500,
              device: str = "cpu", verbose: bool = False) -> SAE:
    set_seed(seed)
    d_in = activations_train.shape[1]
    scale = float(np.sqrt(d_in) / (np.linalg.norm(activations_train, axis=1).mean() + 1e-8))
    x_all = torch.from_numpy(activations_train).float()

    b_dec_init = None
    if use_geometric_median_init:
        sample = x_all[torch.randperm(len(x_all))[: min(4096, len(x_all))]] * scale
        b_dec_init = geometric_median(sample)

    sae = SAE(d_in, d_hidden, method=method, k=k, l1_coef=l1_coef, seed=seed,
              init_bias_geometric_median=b_dec_init).to(device)
    sae.scale = scale
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(seed)
    n = x_all.shape[0]
    k_aux = k_aux or max(1, d_hidden // 8)
    fired = torch.zeros(d_hidden, device=device)

    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        xb = (x_all[idx] * scale).to(device)
        pre = (xb - sae.b_dec) @ sae.W_enc + sae.b_enc
        total, mse, z = sae.loss(xb)
        if use_auxk:
            dead = fired == 0
            total = total + 1.0 / 32 * auxk_loss(sae, xb, sae.decode(z), pre, dead, k_aux)
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sched.step()
        sae.normalize_decoder()
        fired += (z > 0).float().sum(0)

        if (step + 1) % resample_every == 0:
            dead = fired == 0
            if dead.any() and step < steps - resample_every:
                with torch.no_grad():
                    x_hat, _ = sae(xb)
                    err = (xb - x_hat).pow(2).sum(-1)
                    pick = torch.topk(err, min(int(dead.sum()), xb.shape[0])).indices
                    repl = xb[pick] - sae.b_dec
                    repl = repl / (repl.norm(dim=1, keepdim=True) + 1e-8)
                    di = torch.nonzero(dead).squeeze(-1)[: repl.shape[0]]
                    sae.W_dec.data[di] = repl
                    sae.W_enc.data[:, di] = repl.T * 0.1
                    sae.b_enc.data[di] = 0.0
            if verbose:
                print(f"  step {step + 1} mse {mse.item():.4f} dead {int(dead.sum())}")
            fired = torch.zeros(d_hidden, device=device)

    sae.eval()
    return sae


@torch.no_grad()
def sae_encode_all(sae: SAE, activations: np.ndarray, batch: int = 8192, device: str = "cpu") -> np.ndarray:
    outs = []
    for i in range(0, len(activations), batch):
        xb = torch.from_numpy(activations[i:i + batch]).float().to(device) * sae.scale
        outs.append(sae.encode(xb).cpu().numpy().astype(np.float32))
    return np.concatenate(outs, 0)


@torch.no_grad()
def sae_report(sae: SAE, activations: np.ndarray, batch: int = 8192, device: str = "cpu") -> Dict[str, float]:
    """Streaming reconstruction report so large activation matrices are never duplicated."""
    n, d = activations.shape
    s1 = torch.zeros(d, dtype=torch.float64, device=device)
    s2 = torch.zeros((), dtype=torch.float64, device=device)
    sse = torch.zeros((), dtype=torch.float64, device=device)
    l0s, active = [], np.zeros(sae.d_hidden, bool)
    for i in range(0, n, batch):
        xb = torch.from_numpy(activations[i:i + batch]).float().to(device) * sae.scale
        x_hat, z = sae(xb)
        s1 += xb.sum(0).double()
        s2 += xb.pow(2).sum().double()
        sse += (xb - x_hat).pow(2).sum().double()
        l0s.append((z > 0).float().sum(-1).mean().item())
        active |= (z > 0).any(0).cpu().numpy()
    sst = s2 - (s1.pow(2).sum() / n)
    return dict(fvu=float((sse / (sst + 1e-12)).item()), l0=float(np.mean(l0s)),
                dead_frac=float(1 - active.mean()), d_hidden=sae.d_hidden)


def track_convergence(activations_train: np.ndarray, activations_test: np.ndarray, d_hidden: int,
                       budgets: Sequence[int] = (1_164, 3_880, 11_640, 34_920), seeds: Sequence[int] = (0, 1),
                       method: str = "topk", k: int = 32, device: str = "cpu") -> pd.DataFrame:
    """Train `len(seeds)` SAEs at each step budget, report FVU (reconstruction) alongside
    cross-seed MMCS (feature identity) at every budget. The rows this returns are the direct
    evidence for the FVU-vs-MMCS divergence rule in docs/methodology.md — run this on any new
    encoder/layer/dataset before picking a training budget for a stability study, rather than
    reusing the reference budget from a different setup."""
    rows: List[dict] = []
    for steps in budgets:
        saes = [
            train_sae(activations_train, d_hidden=d_hidden, method=method, k=k, seed=s, steps=steps, device=device)
            for s in seeds
        ]
        reports = [sae_report(sae, activations_test, device=device) for sae in saes]
        pair_mmcs = []
        for i in range(len(saes)):
            for j in range(i + 1, len(saes)):
                pair_mmcs.append(dictionary_matching(saes[i], saes[j])["mmcs"])
        rows.append(dict(
            steps=steps,
            fvu=float(np.mean([r["fvu"] for r in reports])),
            dead_frac=float(np.mean([r["dead_frac"] for r in reports])),
            mmcs=float(np.mean(pair_mmcs)) if pair_mmcs else float("nan"),
        ))
    return pd.DataFrame(rows)
