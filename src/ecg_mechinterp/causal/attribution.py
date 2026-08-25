"""Gradient attribution of SAE features to a probe's score — the shortlisting step before
the actual causal test in causal/ablation.py.

Exhaustive ablation (try removing every feature, one at a time, and re-run the full forward
pass) is expensive at dictionary sizes in the thousands. Attribution is the cheap first pass:
one backward pass per batch gives every feature's exact first-order attribution to the probe
score at once, via the chain rule through the SAE encoder. Ablation then verifies the
shortlist for real, since first-order attribution can be wrong when the pathway from feature
to probe score is nonlinear (see docs/methodology.md — attribution and ablation disagreeing is
itself informative about how linear the pathway is, not just noise to average away).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ecg_mechinterp.models.registry import EncoderWrapper
from ecg_mechinterp.sae.model import SAE


class TorchProbe(torch.nn.Module):
    """A fitted sklearn `make_pipeline(StandardScaler, LogisticRegression)` probe, re-expressed
    as a differentiable module so gradients can flow from its score back through the encoder."""

    def __init__(self, pipeline):
        super().__init__()
        scaler = pipeline.steps[0][1]
        est = pipeline.steps[-1][1]
        self.register_buffer("mu", torch.tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(scaler.scale_, dtype=torch.float32))
        w = np.ravel(est.coef_).astype(np.float32)
        b = float(np.ravel(est.intercept_)[0]) if np.size(est.intercept_) else 0.0
        self.register_buffer("w", torch.tensor(w))
        self.register_buffer("b", torch.tensor(b))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return ((emb - self.mu) / self.sd) @ self.w + self.b


def feature_attribution(
    enc: EncoderWrapper, probe: TorchProbe, sae: SAE, layer: str, x_windows: np.ndarray,
    time_pool: int = 8, batch_size: int = 8, device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact first-order attribution of every SAE feature to `probe`'s score, and each
    feature's activation density, over `x_windows` (model-ready input, one row per window).
    Returns (attribution[d_hidden], density[d_hidden])."""
    attr = np.zeros(sae.d_hidden, np.float64)
    dens = np.zeros(sae.d_hidden, np.float64)
    n_seen = 0
    for i in tqdm(range(0, len(x_windows), batch_size), desc="attribution", leave=False):
        xb = torch.from_numpy(x_windows[i:i + batch_size]).to(device).requires_grad_(True)
        emb, store = enc(xb, capture=[layer], detach=False)
        h = store[layer]
        score = probe(emb).sum()
        g = torch.autograd.grad(score, h, retain_graph=False)[0]  # (B, C, T)
        with torch.no_grad():
            p = time_pool
            gp = F.avg_pool1d(g, p, p) * p if p > 1 else g
            hp = F.avg_pool1d(h.detach(), p, p) if p > 1 else h.detach()
            z = sae.encode(hp.transpose(1, 2) * sae.scale)  # (B, Tp, F)
            gd = torch.einsum("bct,fc->btf", gp, sae.W_dec)  # (B, Tp, F)
            attr += (z * gd).sum((0, 1)).double().cpu().numpy() / sae.scale
            dens += (z > 0).float().sum((0, 1)).double().cpu().numpy()
            n_seen += z.shape[0] * z.shape[1]
    return attr, dens / max(1, n_seen)
