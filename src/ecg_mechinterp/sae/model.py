"""Sparse autoencoder for decomposing layer activations into sparse features.

Combines the RQ1-4 notebook's SAE (TopK or L1-ReLU, cosine LR schedule, dead-feature
resampling from high-reconstruction-error tokens) with two refinements from the CLEF SAE
convergence study that were specifically validated to matter for the *stability* measurements
downstream (see docs/provenance.md), both opt-in via constructor flags so this stays a
superset, not a rewrite, of either:

- `init_bias_geometric_median`: the pre-encoder bias initialized to the geometric median of a
  sample of activations, rather than zero — a more robust center than the mean when
  activations have outliers.
- `use_auxk`: an auxiliary loss letting dead features reconstruct the residual that live
  features miss, which the convergence study introduced specifically because features that
  stop firing early never recover on their own, inflating apparent cross-seed instability for
  reasons that have nothing to do with the seed.

TopK is the default method (exact L0 control, no shrinkage on surviving coefficients); L1 is
kept for the method-sensitivity check in sae/stability.py (do stable features depend on the
training objective, not just the seed?).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def geometric_median(x: torch.Tensor, iters: int = 100, tol: float = 1e-7) -> torch.Tensor:
    """Weiszfeld's algorithm. x: (n, d)."""
    y = x.mean(0)
    for _ in range(iters):
        d = torch.norm(x - y, dim=1).clamp_min(1e-8)
        w = 1.0 / d
        y_new = (w[:, None] * x).sum(0) / w.sum()
        if torch.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y


class SAE(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, method: str = "topk", k: int = 32,
                 l1_coef: float = 4e-3, seed: int = 0,
                 init_bias_geometric_median: Optional[torch.Tensor] = None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.d_in, self.d_hidden, self.method, self.k, self.l1_coef = d_in, d_hidden, method, k, l1_coef
        w = torch.randn(d_hidden, d_in, generator=g) / math.sqrt(d_in)
        w = w / w.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(w.clone())  # (F, d) -- tied init: W_enc = W_dec.T
        self.W_enc = nn.Parameter(w.clone().T.contiguous())
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        b_dec_init = (init_bias_geometric_median.clone() if init_bias_geometric_median is not None
                      else torch.zeros(d_in))
        self.b_dec = nn.Parameter(b_dec_init)
        self.scale = 1.0  # set by train_sae to the activation-norm rescaling factor

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.method == "topk":
            v, i = torch.topk(pre, self.k, dim=-1)
            z = torch.zeros_like(pre).scatter_(-1, i, F.relu(v))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    def loss(self, x: torch.Tensor):
        x_hat, z = self(x)
        mse = (x_hat - x).pow(2).sum(-1).mean()
        total = mse
        if self.method == "l1":
            total = total + self.l1_coef * (z.abs() * self.W_dec.norm(dim=1)[None, :]).sum(-1).mean()
        return total, mse, z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-8)

    def decoder_directions(self) -> np.ndarray:
        """(dict_size, input_dim) — one row per learned feature, for stability matching."""
        return self.W_dec.detach().cpu().numpy()


def auxk_loss(sae: SAE, x: torch.Tensor, x_hat: torch.Tensor, pre: torch.Tensor,
              dead_mask: torch.Tensor, k_aux: int) -> torch.Tensor:
    """Let dead features reconstruct the residual that live features miss (CLEF SAE
    convergence study, section 5). Without this, features that stop firing never recover, and
    the dead fraction inflates apparent cross-seed instability for reasons unrelated to the
    seed. `pre` is the pre-topk encoder activation; `dead_mask` (d_hidden,) bool."""
    n_dead = int(dead_mask.sum())
    if n_dead == 0:
        return torch.zeros((), device=x.device)
    k_aux = min(k_aux, n_dead)
    pre_dead = pre.masked_fill(~dead_mask, float("-inf"))
    vals, idx = torch.topk(pre_dead, k_aux, dim=-1)
    z_aux = torch.zeros_like(pre).scatter_(-1, idx, vals)
    residual = x - x_hat.detach()
    return ((z_aux @ sae.W_dec - residual) ** 2).sum(-1).mean()


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance unexplained, the standard SAE reconstruction metric. Read this
    together with sae/train.py's `track_convergence` before treating a low FVU as evidence
    the SAE has converged — see docs/methodology.md's EV/FVU-vs-feature-identity rule."""
    return ((x - x_hat).pow(2).sum() / ((x - x.mean(0, keepdim=True)).pow(2).sum() + 1e-12)).item()
