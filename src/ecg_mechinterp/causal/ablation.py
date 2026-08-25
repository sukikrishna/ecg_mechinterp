"""Feature ablation: the actual causal test, verifying what attribution.py only shortlisted.

`ablation_effect` measures the change in a probe's score/AUROC when a set of SAE features is
removed at the hooked layer, exactly (not approximately) under average pooling — the SAE reads
pooled activations `hp = avgpool(h, p)`, so removing feature i means subtracting `z_i @ d_i`
from `hp`, achieved on the unpooled `h` by subtracting that same quantity broadcast across
each pooling window.

`group_ablation` is the piece that turned a correlational finding (docs/findings.md: MI
decoding correlates with age/sex in this cohort) into a causal one: ablating the features
most attributed to a *different* probe (e.g. sex) and re-measuring a *target* probe (e.g. MI),
against a density-matched random control so the effect isn't just "removing some features
drops accuracy" — see bias/demographic.py, which is this function applied to that specific
comparison.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ecg_mechinterp.causal.attribution import TorchProbe
from ecg_mechinterp.models.registry import EncoderWrapper
from ecg_mechinterp.sae.model import SAE


def make_feature_editor(sae: SAE, feature_ids: Sequence[int], mode: str = "ablate",
                         alpha: float = 1.0, time_pool: int = 8):
    """Return an `edit_fn` for `EncoderWrapper.forward(..., edit_fn=..., edit_layer=...)` that
    removes (`mode="ablate"`) or rescales (`mode="scale"`, by `alpha`) the listed features'
    contribution."""
    fid = torch.as_tensor(list(feature_ids), dtype=torch.long)

    def edit_fn(h: torch.Tensor) -> torch.Tensor:  # h: (B, C, T)
        fid_ = fid.to(h.device)
        hp = F.avg_pool1d(h, time_pool, time_pool) if time_pool > 1 else h
        xt = hp.transpose(1, 2) * sae.scale  # (B, Tp, C)
        z = sae.encode(xt)
        d = sae.W_dec[fid_]  # (n_feat, C)
        contrib = z[..., fid_] @ d  # (B, Tp, C)
        delta = contrib.transpose(1, 2) / sae.scale
        if time_pool > 1:
            delta = delta.repeat_interleave(time_pool, dim=-1)
        if delta.shape[-1] < h.shape[-1]:
            delta = F.pad(delta, (0, h.shape[-1] - delta.shape[-1]))
        delta = delta[..., : h.shape[-1]]
        return h - delta if mode == "ablate" else h + (alpha - 1.0) * delta

    return edit_fn


@torch.no_grad()
def ablation_effect(
    enc: EncoderWrapper, probe: TorchProbe, sae: SAE, layer: str, feature_ids: Sequence[int],
    x_windows: np.ndarray, y: np.ndarray, time_pool: int = 8, batch_size: int = 16,
    device: str = "cpu",
) -> Dict[str, float]:
    """Change in probe score (and AUROC, for a binary concept) when `feature_ids` are
    removed at `layer`, over `x_windows`/`y`."""
    base, abl = [], []
    editor = make_feature_editor(sae, feature_ids, mode="ablate", time_pool=time_pool)
    for i in range(0, len(x_windows), batch_size):
        xb = torch.from_numpy(x_windows[i:i + batch_size]).to(device)
        e0, _ = enc(xb)
        base.append(probe(e0).cpu().numpy())
        e1, _ = enc(xb, edit_fn=editor, edit_layer=layer)
        abl.append(probe(e1).cpu().numpy())
    base, abl = np.concatenate(base), np.concatenate(abl)
    out = dict(delta_score=float(np.mean(np.abs(abl - base))), signed_delta=float(np.mean(abl - base)))
    if len(np.unique(y)) > 1:
        out["auroc_base"] = float(roc_auc_score(y, base))
        out["auroc_ablated"] = float(roc_auc_score(y, abl))
        out["auroc_drop"] = out["auroc_base"] - out["auroc_ablated"]
    return out


def density_matched_control(feature_ids: Sequence[int], density: np.ndarray, seed: int = 0) -> list:
    """A same-size feature set, matched on activation density to `feature_ids` but disjoint
    from it — the control every ablation in this package should be compared against, so a
    drop isn't just "removing some active features hurts accuracy" (see module docstring)."""
    alive = np.flatnonzero(density > 1e-4)
    pool = [f for f in alive if f not in set(feature_ids)]
    control = []
    for target_density in density[list(feature_ids)]:
        j = int(np.argmin(np.abs(density[pool] - target_density)))
        control.append(pool.pop(j))
    return control
