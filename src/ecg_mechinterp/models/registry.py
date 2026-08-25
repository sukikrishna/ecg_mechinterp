"""Encoder registry: resolves a config's `encoders` names to loaded, hook-ready models.

Generalizes the stability notebook's `ENCODERS`/`build_encoder()` pattern (which only knew
about CLEF's three sizes plus its own from-scratch reference architecture) to every model in
`ecg_mechinterp.models` and its dataset-verified `layer_names`, instead of the notebook's
auto-discovery heuristic ("the deepest ModuleList holding >= 3 children"). Auto-discovery was
the right tool when the notebook couldn't assume anything about an unknown repo's module
tree; here every wrapped model already declares verified `layer_names` (see docs/models.md),
so hooking by name is strictly safer than re-deriving depth structure at runtime.

Naming convention (kept identical to the notebook and to config.py's docstring): a "_rand"
suffix on an encoder key requests that encoder's architecture-matched random-initialization
control instead of its pretrained weights. Always include at least one control before trusting
a raw similarity number — see docs/methodology.md.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ecg_mechinterp.models.base import ECGModel
from ecg_mechinterp.models.clef import CLEF
from ecg_mechinterp.models.ecgfounder import ECGFounder
from ecg_mechinterp.models.ecgjepa import ECGJEPA

# encoder key -> (builder, weights path under cfg.ckpt_dir, supports "_rand"). Paths match the
# filenames scripts/setup_*.sh actually write (e.g. weights/ecgfounder/12_lead_ECGFounder.pth
# from Hugging Face, weights/clef/clef_medium.ckpt from Zenodo) — keep these two in sync if a
# setup script's destination filename ever changes.
_BUILDERS = {
    "ecgfounder_12lead": (lambda: ECGFounder(leads=12), "ecgfounder/12_lead_ECGFounder.pth", True),
    "ecgfounder_1lead": (lambda: ECGFounder(leads=1), "ecgfounder/1_lead_ECGFounder.pth", True),
    "clef_s": (lambda: CLEF(size="small"), "clef/clef_small.ckpt", True),
    "clef_m": (lambda: CLEF(size="medium"), "clef/clef_medium.ckpt", True),
    "clef_l": (lambda: CLEF(size="large"), "clef/clef_largel.ckpt", True),
    "ecgjepa": (lambda: ECGJEPA(), "ecgjepa/multiblock_epoch100.pth", False),
}


def _base_key(key: str) -> str:
    return key[: -len("_rand")] if key.endswith("_rand") else key


def build_model(key: str, ckpt_dir: str, seed: int = 0) -> Tuple[ECGModel, str]:
    """Build and load one ECGModel by registry key. Returns (model, info string)."""
    base = _base_key(key)
    if base not in _BUILDERS:
        raise KeyError(f"unknown encoder {key!r}; known: {sorted(_BUILDERS)} (+ '_rand' suffix)")
    factory, fname, supports_rand = _BUILDERS[base]
    model = factory()
    if key.endswith("_rand"):
        if not supports_rand:
            raise NotImplementedError(f"{base} has no random-init control yet (see its module docstring)")
        # Deterministic per-key seed so re-running the same config reproduces the same control.
        derived_seed = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % (2 ** 31)
        model.load_random(seed=derived_seed if seed == 0 else seed)
        return model, f"{base}, random initialization, seed={derived_seed}"
    weights_path = os.path.join(ckpt_dir, fname)
    model.load(weights_path)
    return model, f"{base}, pretrained weights from {weights_path}"


class EncoderWrapper(nn.Module):
    """Uniform interface over any ECGModel: hooked forward with activation capture and
    causal-intervention support (`edit_fn` at `edit_layer`), built on the model's own
    verified `layer_names` rather than auto-discovered module structure."""

    def __init__(self, ecg_model: ECGModel, name: str, info: str = ""):
        super().__init__()
        self.ecg_model = ecg_model
        self.model = ecg_model.model  # the underlying nn.Module, for named_modules() lookup
        self.name = name
        self.info = info
        for p in self.model.parameters():
            p.requires_grad_(False)
        modules = dict(self.model.named_modules())
        missing = [n for n in ecg_model.layer_names if n not in modules]
        if missing:
            raise KeyError(
                f"{name}: layer_names {missing} not found in named_modules(); "
                f"available include: {sorted(modules)[:20]}"
            )
        self.hook_points: List[Tuple[str, nn.Module]] = [(n, modules[n]) for n in ecg_model.layer_names]

    def layer_names(self) -> List[str]:
        return [n for n, _ in self.hook_points]

    def layers_at_relative_depth(self, depths: Sequence[float]) -> Dict[float, str]:
        names = self.layer_names()
        out = {}
        for d in depths:
            i = min(len(names) - 1, max(0, int(round(d * len(names))) - 1))
            out[d] = names[i]
        return out

    def forward(self, x: torch.Tensor, capture: Optional[Sequence[str]] = None,
                edit_fn=None, edit_layer: Optional[str] = None, detach: bool = True):
        """Run the encoder via the wrapped model's own `preprocess`-free forward (`x` is
        already model-ready). `capture` collects raw stage outputs (B, C, T); `edit_fn`
        replaces the output of `edit_layer` in place, which is how feature ablation/scaling
        is applied (see causal/ablation.py). Set `detach=False` for gradient attribution.

        The returned pooled `emb` is always the *last hooked layer's* activation, time-pooled
        — never `self.ecg_model.forward(x)`'s raw return value. That distinction matters:
        CLEF's pretrained loader swaps its classification head for `nn.Identity`, so its raw
        forward() output happens to already be the pooled backbone feature, but ECGFounder's
        wrapper does not (see docs/models.md) — using raw forward() output as "the embedding"
        would silently probe on 150-way classification logits for ECGFounder while probing on
        the real backbone feature for CLEF. Deriving `emb` from the last hook point instead is
        correct for every model regardless of whether its head was swapped.
        """
        last_layer = self.hook_points[-1][0]
        store: Dict[str, torch.Tensor] = {}
        last_out = {}
        handles = []
        wanted = set(capture or [])
        need_hook = wanted | {last_layer, edit_layer} - {None}

        def mk_hook(nm):
            def hook(_m, _i, out):
                if isinstance(out, tuple):
                    out = out[0]
                if nm in wanted:
                    store[nm] = out.detach() if detach else out
                if nm == last_layer:
                    last_out["emb"] = out.detach() if detach else out
                if edit_fn is not None and nm == edit_layer:
                    return edit_fn(out)
                return None
            return hook

        for nm, mod in self.hook_points:
            if nm in need_hook:
                handles.append(mod.register_forward_hook(mk_hook(nm)))
        try:
            self.ecg_model.forward(x)
        finally:
            for h in handles:
                h.remove()
        emb = last_out["emb"]
        if emb.dim() == 3:
            emb = emb.mean(-1)
        return emb, store


def build_encoder_registry(keys: Sequence[str], ckpt_dir: str, device: str = "cpu",
                            seed: int = 0) -> Tuple[Dict[str, EncoderWrapper], Dict[str, str]]:
    """Build every encoder in `keys`. Returns (loaded, failed) so a run can proceed on
    whatever loaded rather than aborting entirely on one missing checkpoint/repo clone."""
    loaded: Dict[str, EncoderWrapper] = {}
    failed: Dict[str, str] = {}
    for key in keys:
        try:
            model, info = build_model(key, ckpt_dir, seed=seed)
            enc = EncoderWrapper(model, key, info).to(torch.device(device)).eval()
            loaded[key] = enc
            n_params = sum(p.numel() for p in enc.model.parameters())
            print(f"[{key}] {info} | params {n_params / 1e6:.2f}M | stages {len(enc.hook_points)}")
        except Exception as e:  # noqa: BLE001 - collected and reported, not swallowed
            failed[key] = str(e)
            print(f"[{key}] FAILED\n{e}\n")
    if not loaded:
        raise RuntimeError(
            "No encoder loaded; everything downstream depends on at least one. "
            f"Failures: {failed}"
        )
    return loaded, failed
