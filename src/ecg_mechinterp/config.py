"""Central configuration for a mechinterp run.

Generalizes the notebook's `Config` dataclass (which was CLEF + PTB-DB specific) to any
combination of dataset and encoder registry entries, so the same RQ1-4 pipeline runs on
PTB-XL/multi-model as well as PTB-DB/CLEF-only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple


@dataclass
class Config:
    # paths
    root: str = "."
    data_dir: str = "data"
    ckpt_dir: str = "weights"
    cache_dir: str = "cache"
    out_dir: str = "results"

    # dataset selection: "ptbxl" or "ptbdb"
    dataset: str = "ptbxl"
    lead: str = "i"  # single-lead models read this lead when the dataset has 12
    max_records: Optional[int] = None  # cap for smoke tests

    # signal
    target_fs: int = 500
    window_sec: float = 10.0
    stride_sec: float = 5.0
    max_windows_per_record: int = 12

    # encoders: names must resolve via ecg_mechinterp.models.registry.build_encoder.
    # A "_rand" suffix (e.g. "clef_m_rand") requests the architecture-matched random-init
    # control for that base encoder — always include at least one when the analysis will
    # make any claim from a raw CKA/similarity number (see docs/methodology.md).
    encoders: Tuple[str, ...] = ("ecgfounder_1lead", "clef_m", "ecgjepa")
    relative_depths: Tuple[float, ...] = (0.5, 0.75, 1.0)
    batch_size: int = 16

    # activation extraction
    time_pool: int = 8
    tokens_per_window: int = 32
    match_tokens: int = 12000

    # SAE
    sae_expansions: Tuple[int, ...] = (4,)
    sae_max_hidden: int = 4096
    sae_k: Tuple[int, ...] = (32,)
    sae_methods: Tuple[str, ...] = ("topk",)
    sae_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    sae_steps: int = 34_920  # see docs/methodology.md: this is the training budget at which
    # the reference SAE convergence study (12/40/120/360 epochs on ~2,900 windows) found seed
    # agreement still rising, not the number that made EV plateau. Do not lower this without
    # rerunning the EV-vs-MMCS divergence check in sae.train.track_convergence first.
    sae_batch: int = 4096
    sae_lr: float = 3e-4
    sae_l1_coef: float = 4e-3

    # analysis
    n_bootstrap: int = 500
    n_permutation: int = 200
    knn_k: int = 20
    causal_top_features: int = 24
    causal_eval_windows: int = 384
    stable_rank_threshold: float = 0.7

    seed: int = 0
    device: str = "cpu"

    def ensure_dirs(self) -> None:
        for d in [self.data_dir, self.ckpt_dir, self.cache_dir, self.out_dir,
                  os.path.join(self.out_dir, "figures"), os.path.join(self.out_dir, "tables")]:
            os.makedirs(d, exist_ok=True)

    @property
    def window_len(self) -> int:
        return int(self.window_sec * self.target_fs)
