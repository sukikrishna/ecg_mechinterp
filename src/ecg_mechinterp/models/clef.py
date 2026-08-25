"""Wrapper around Nokia Bell Labs' CLEF (the Net1D architecture, single-lead).

CLEF isn't published as a pip package, so scripts/setup_clef.sh clones its source into
external/ecg-foundation-model and this wrapper imports from there for pretrained loading.

Important caveat for cross-model comparisons (see docs/models.md): CLEF-medium and
ECGFounder both build the exact same Net1D configuration (identical filter_list/
m_blocks_list at every stage) — they are the same backbone architecture trained on
different data/objectives, not two independently designed architectures. Any shared
representation found between CLEF and ECGFounder is evidence about shared training
data/objective effects, not architecture-independence — ECG-JEPA is the model that actually
lets an architecture-independence claim be tested (see ecgjepa.py).

Random-initialization control (`pretrained=False`): built by instantiating the same net1d.Net1D
class directly with CLEF's published architecture config (base_filters/filter_list/
m_blocks_list per size, from the CLEF paper Appendix C.3 Table S7), skipping the external
repo's `create_net1d_by_size` factory entirely. This is not a guess: it is the same
direct-instantiation approach the CLEF SAE convergence study used for clef_m specifically
(vendor/net1d.py there), and it was cross-validated there against the real checkpoint via a
strict state_dict load, an exact parameter-count assertion, and a "weights actually changed
after load" check — see docs/provenance.md. The random-init path below reuses that same,
already-verified Net1D class purely for its architecture shape; it is never used to hold
pretrained weights, matching the discipline the CLEF SAE study and the stability notebook
both used for their random-init controls.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from ecg_mechinterp.models.base import ECGModel

_CLEF_REPO = Path(__file__).resolve().parents[3] / "external" / "ecg-foundation-model"
if str(_CLEF_REPO) not in sys.path:
    sys.path.insert(0, str(_CLEF_REPO))

# Per-size Net1D configs (CLEF paper Appendix C.3 Table S7). clef_m's numbers are the ones
# independently confirmed against the released checkpoint's own key shapes and parameter
# count (30,654,000) in the CLEF SAE convergence study; small/large follow the same table.
_NET1D_CONFIGS = {
    "small": dict(
        base_filters=32, ratio=0.5, groups_width=8,
        filter_list=[32, 64, 64, 128, 128, 256],
        m_blocks_list=[1, 1, 2, 2, 2, 2],
    ),
    "medium": dict(
        base_filters=64, ratio=1.0, groups_width=16,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    ),
    "large": dict(
        base_filters=128, ratio=1.5, groups_width=32,
        filter_list=[128, 256, 256, 512, 512, 1024, 1024, 2048, 2048],
        m_blocks_list=[2, 3, 3, 4, 4, 5, 5, 6, 6],
    ),
}
_N_STAGES = {"small": 6, "medium": 7, "large": 9}


class CLEF(ECGModel):
    """Single-lead CLEF (Net1D + SE blocks), small/medium/large. Pretrained preprocessing
    and hook points verified against the original repo's CLEF.py / contrastive_dataloader.py
    — see docs/models.md."""

    INPUT_LENGTH = 5000  # samples
    SAMPLING_RATE = 500  # Hz

    def __init__(self, size: str = "medium"):
        if size not in _NET1D_CONFIGS:
            raise ValueError("CLEF ships small/medium/large checkpoints")
        self.size = size
        self.model = None  # built in load()/load_random()

    def load(self, weights_path: str) -> None:
        try:
            from clef.baselines.models.CLEF import create_net1d_by_size
        except ImportError as e:
            raise ImportError(
                "CLEF source not found. Run scripts/setup_clef.sh first to clone "
                "github.com/Nokia-Bell-Labs/ecg-foundation-model into "
                "external/ecg-foundation-model."
            ) from e
        # n_classes is irrelevant here: create_net1d_by_size replaces the head with
        # nn.Identity, so forward() returns the pooled backbone feature, not logits.
        self.model = create_net1d_by_size(
            device=torch.device("cpu"),
            model_size=self.size,
            n_classes=4,
            linear_prob=False,
            pth=weights_path,
            in_channels=1,
        )
        self.model.eval()

    def load_random(self, seed: int = 0) -> None:
        """Architecture-matched random-initialization control — see module docstring for why
        this bypasses the pretrained-loading factory entirely."""
        from net1d import Net1D  # vendored/cloned alongside ECGFounder; same class

        torch.manual_seed(seed)
        cfg = _NET1D_CONFIGS[self.size]
        model = Net1D(
            in_channels=1,
            base_filters=cfg["base_filters"],
            ratio=cfg["ratio"],
            filter_list=cfg["filter_list"],
            m_blocks_list=cfg["m_blocks_list"],
            kernel_size=16,
            stride=2,
            groups_width=cfg["groups_width"],
            n_classes=1,
            use_bn=False,
            use_do=False,
        )
        model.dense = nn.Identity()
        model.eval()
        self.model = model

    def preprocess(self, signal: np.ndarray, lead: int = 0) -> torch.Tensor:
        """`signal`: (samples, leads) in standard clinical lead order. CLEF takes a single
        lead — Lead I (index 0) by default, chosen to match ECGFounder's 1-lead checkpoint
        (trained specifically on Lead I) so both models can be fed the literal same channel
        for a fair comparison; CLEF itself is roughly lead-invariant since pretraining
        randomly picked a lead per example.

        Resamples to 5000 samples (500Hz, 10s), applies a 0.67-40Hz Butterworth bandpass,
        then a per-sample z-score — CLEF's own pretraining path
        (clef/data/contrastive_dataloader.py), not its PTB-XL downstream loader (which uses
        sklearn StandardScaler instead); the pretraining path is the more faithful choice
        for representation extraction.
        """
        x = np.asarray(signal[:, lead], dtype=np.float64)
        if len(x) != self.INPUT_LENGTH:
            x_old = np.linspace(0, 1, len(x))
            x_new = np.linspace(0, 1, self.INPUT_LENGTH)
            x = np.interp(x_new, x_old, x)
        sos = scipy.signal.butter(4, [0.67, 40], btype="bandpass", fs=self.SAMPLING_RATE, output="sos")
        x = scipy.signal.sosfiltfilt(sos, x)
        x = (x - x.mean()) / (x.std() + 1e-8)
        return torch.from_numpy(x.copy()).float().view(1, 1, -1)  # (1, 1, 5000)

    @property
    def layer_names(self) -> List[str]:
        return ["first_conv"] + [f"stage_list.{i}" for i in range(_N_STAGES[self.size])]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
