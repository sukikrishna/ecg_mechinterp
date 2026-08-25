"""Wrapper around PKUDigitalHealth/ECGFounder (the Net1D architecture).

ECGFounder ships as a plain 1D CNN (net1d.Net1D) — not a transformer despite the name — and
isn't published as a pip package, so scripts/setup_ecgfounder.sh clones its source into
external/ECGFounder and this wrapper imports from there.

Ported unchanged from the ecg_interp repo (src/ecg_interp/models/ecgfounder.py). Its __init__
already builds the architecture separately from load() loading weights, so
`ECGFounder(leads=...)` without calling `.load()` is already a valid random-initialization
control — no extra code needed, unlike CLEF/ECG-JEPA below (see clef.py, ecgjepa.py, and
models/registry.py).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

from ecg_mechinterp.models.base import ECGModel

_ECGFOUNDER_REPO = Path(__file__).resolve().parents[3] / "external" / "ECGFounder"
if str(_ECGFOUNDER_REPO) not in sys.path:
    sys.path.insert(0, str(_ECGFOUNDER_REPO))


class ECGFounder(ECGModel):
    """12-lead or 1-lead ECGFounder. Preprocessing and hook points verified against the
    original repo's net1d.py / dataset.py / ptbxl_eval.py — see docs/models.md."""

    # Identical between the 12-lead and 1-lead checkpoints except in_channels.
    _NET1D_KWARGS = dict(
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes=150,
        use_bn=False,  # required to match how the released checkpoint was trained/evaluated
        use_do=False,  # (BN/dropout submodules exist in the class but must stay unused)
    )
    INPUT_LENGTH = 5000  # samples
    SAMPLING_RATE = 500  # Hz

    def __init__(self, leads: int = 12):
        if leads not in (1, 12):
            raise ValueError("ECGFounder only ships 1-lead and 12-lead checkpoints")
        self.leads = leads
        try:
            from net1d import Net1D
        except ImportError as e:
            raise ImportError(
                "ECGFounder source not found. Run scripts/setup_ecgfounder.sh first to clone "
                "github.com/PKUDigitalHealth/ECGFounder into external/ECGFounder."
            ) from e
        self.model = Net1D(in_channels=leads, **self._NET1D_KWARGS)
        self.model.eval()

    def load(self, weights_path: str) -> None:
        # weights_only=False: this checkpoint predates torch's safer default and pickles a
        # numpy scalar; fine here since it's the official PKUDigitalHealth/ECGFounder release.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        """`signal`: (samples, leads) in standard clinical lead order (I, II, III, aVR, aVL,
        aVF, V1-V6), as PTB-XL stores it. Resamples to 5000 samples (500Hz, 10s) and applies
        ECGFounder's global (not per-channel) z-score, matching its dataset.py /
        ptbxl_eval.py baseline path (no bandpass filter — that's only used in their
        fine-tuning scripts, not the base evaluation path this wrapper follows)."""
        signal = np.asarray(signal, dtype=np.float32)
        if signal.shape[0] != self.INPUT_LENGTH:
            x_old = np.linspace(0, 1, signal.shape[0])
            x_new = np.linspace(0, 1, self.INPUT_LENGTH)
            signal = np.stack(
                [np.interp(x_new, x_old, signal[:, lead]) for lead in range(signal.shape[1])],
                axis=1,
            )
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        return torch.from_numpy(signal.T).float().unsqueeze(0)  # (1, leads, 5000)

    @property
    def layer_names(self) -> List[str]:
        return ["first_conv"] + [f"stage_list.{i}" for i in range(7)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
