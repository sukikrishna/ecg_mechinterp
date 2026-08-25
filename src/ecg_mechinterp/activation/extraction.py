"""Activation extraction: pooled per-window embeddings for probing, and token-level
activations at matched relative positions for SAE/similarity work across encoders.

The token-position logic is the RQ1-4 notebook's: positions are drawn once, as *fractions* of
the sequence length rather than absolute indices, and reused for every encoder. Two encoders
at the same relative depth can have very different temporal resolution (a CNN's stage 3 may be
157 samples long, a transformer's block 3 several times that) — sampling by fraction is what
makes "token j of window w" refer to the same moment in the same heartbeat everywhere, which a
fixed absolute index would not.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ecg_mechinterp.models.registry import EncoderWrapper


def token_fractions(n_tokens: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.random(n_tokens))


def positions_for_length(fractions: np.ndarray, n_pos: int) -> np.ndarray:
    pos = np.floor(fractions * n_pos).astype(int)
    return np.clip(pos, 0, n_pos - 1)


@torch.no_grad()
def extract_pooled_embedding(enc: EncoderWrapper, x_batch: np.ndarray,
                              batch_size: int = 16, device: str = "cpu") -> np.ndarray:
    """(N, ...) model-ready input -> (N, D) pooled embedding, for linear probing.

    Always the *last* hooked layer's activation, full-time-average pooled (see
    `EncoderWrapper.forward`'s docstring on why this — not the model's raw `forward()` output
    — is the correct model-agnostic definition of "the embedding")."""
    outs = []
    for i in range(0, len(x_batch), batch_size):
        xb = torch.from_numpy(x_batch[i:i + batch_size]).to(device)
        emb, _ = enc(xb)
        outs.append(emb.float().cpu().numpy())
    return np.concatenate(outs, 0)


@torch.no_grad()
def extract_layer_pooled(enc: EncoderWrapper, x_batch: np.ndarray, layer: str,
                          batch_size: int = 16, device: str = "cpu") -> np.ndarray:
    """(N, ...) model-ready input -> (N, C) full-time-average-pooled activation at a specific
    named layer (not necessarily the last one) — what a depth profile needs at every layer,
    as opposed to `extract_pooled_embedding`'s always-last-layer convention or
    `extract_token_activations`'s sparse multi-token sampling (built for SAE/similarity work
    that specifically needs many tokens per window, not one pooled summary)."""
    outs = []
    for i in range(0, len(x_batch), batch_size):
        xb = torch.from_numpy(x_batch[i:i + batch_size]).to(device)
        _, store = enc(xb, capture=[layer])
        h = store[layer]  # (B, C, T)
        outs.append(h.float().mean(-1).cpu().numpy())
    return np.concatenate(outs, 0)


@torch.no_grad()
def extract_token_activations(
    enc: EncoderWrapper, x_batch: np.ndarray, layers: Sequence[str], fractions: np.ndarray,
    time_pool: int = 8, batch_size: int = 16, device: str = "cpu", desc: str = "",
) -> Dict[str, np.ndarray]:
    """Return {layer: (N, len(fractions), C) float16} plus '__emb__' -> (N, D) float32.

    Activations are average-pooled by `time_pool` along time before sampling positions, so
    the kept tokens summarize a short window of timesteps rather than a single instant —
    matches the RQ1-4 notebook's extraction, which found this necessary for stable per-token
    correlation metrics.
    """
    outs = {nm: [] for nm in layers}
    embs = []
    for i in tqdm(range(0, len(x_batch), batch_size), desc=f"acts:{desc or enc.name}", leave=False):
        xb = torch.from_numpy(x_batch[i:i + batch_size]).to(device)
        emb, store = enc(xb, capture=list(layers))
        embs.append(emb.float().cpu().numpy())
        for nm in layers:
            h = store[nm]  # (B, C, T)
            if time_pool > 1 and h.shape[-1] >= time_pool:
                h = F.avg_pool1d(h, time_pool, time_pool)
            pos = torch.as_tensor(positions_for_length(fractions, h.shape[-1]), device=h.device)
            h = h.index_select(-1, pos).transpose(1, 2)  # (B, T_kept, C)
            outs[nm].append(h.to(torch.float16).cpu().numpy())
    res = {nm: np.concatenate(v, 0) for nm, v in outs.items()}
    res["__emb__"] = np.concatenate(embs, 0)
    res["__fractions__"] = fractions
    return res


def flat_tokens(activations: np.ndarray, window_idx: np.ndarray) -> np.ndarray:
    """(N, T, C) token activations for a subset of windows -> (len(window_idx) * T, C) float32,
    flattening the token axis into extra rows. The standard shape every similarity/SAE
    function in this package expects."""
    return activations[window_idx].reshape(-1, activations.shape[-1]).astype(np.float32)


class ActivationExtractor:
    """Simple forward-hook capture for a single pass, kept from the ecg_interp repo for
    one-off inspection (e.g. verifying a model's layer shapes) where the batched, token-level
    machinery above is more than is needed.

    Usage:
        with ActivationExtractor(model, ["layer1", "layer2.0"]) as extractor:
            model(x)
        activations = extractor.activations  # {name: tensor}
    """

    def __init__(self, model: torch.nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self) -> "ActivationExtractor":
        modules = dict(self.model.named_modules())
        for name in self.layer_names:
            if name not in modules:
                raise KeyError(
                    f"'{name}' is not a module of this model. "
                    f"Available names include: {sorted(modules)[:20]}"
                )
            handle = modules[name].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            self.activations[name] = output.detach().cpu()

        return hook
