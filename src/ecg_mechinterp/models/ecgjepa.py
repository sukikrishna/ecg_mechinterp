"""Wrapper around ECG-JEPA (github.com/sehunfromdaegu/ECG_JEPA) — a genuine transformer/JEPA
architecture (multi-head attention blocks, no convolutions anywhere), unlike ECGFounder and CLEF
which both turn out to be the same `Net1D` CNN backbone (see docs/models.md). This is the model
that actually lets an architecture-independence claim be tested.

Not published as a pip package, so scripts/setup_ecgjepa.sh clones its source into
external/ecg-jepa and this wrapper imports from there.

Ported unchanged from the ecg_interp repo (src/ecg_interp/models/ecgjepa.py). Unlike CLEF, no
random-initialization control is implemented here yet: `load_encoder()` conflates building the
transformer with loading its weights, and reproducing its exact hyperparameters (embedding
dim, depth, heads, masking config) independently would mean guessing at an architecture this
wrapper has not verified — the thing docs/models.md and CLAUDE.md both warn against. Wiring a
`load_random()` here is real, scoped follow-up work (see docs/provenance.md), not something to
approximate.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import scipy.signal
import torch

from ecg_mechinterp.models.base import ECGModel

_ECGJEPA_REPO = Path(__file__).resolve().parents[3] / "external" / "ecg-jepa"
if str(_ECGJEPA_REPO) not in sys.path:
    sys.path.insert(0, str(_ECGJEPA_REPO))

# Indices into PTB-XL's standard 12-lead order [I,II,III,aVR,aVL,aVF,V1..V6] that give
# ECG-JEPA's expected 8 leads [I,II,V1..V6] (III/aVR/aVL/aVF are dropped -- linearly derivable
# from I & II, so ECG-JEPA doesn't use them).
PTBXL_LEAD_INDICES_FOR_ECGJEPA = [0, 1, 6, 7, 8, 9, 10, 11]


class ECGJEPA(ECGModel):
    """Multi-block-masking checkpoint by default (the repo's own default and the only variant
    with a shipped eval log). Preprocessing and hook points verified against the original
    repo's ecg_jepa.py / models.py — see docs/models.md."""

    INPUT_LENGTH = 2500  # samples
    SAMPLING_RATE = 250  # Hz (effective, after resampling)

    def __init__(self):
        self.model = None  # built in load(); ECG-JEPA's own loader conflates instantiate+load

    def load(self, weights_path: str) -> None:
        try:
            from models import load_encoder
        except ImportError as e:
            raise ImportError(
                "ECG-JEPA source not found. Run scripts/setup_ecgjepa.sh first to clone "
                "github.com/sehunfromdaegu/ECG_JEPA into external/ecg-jepa."
            ) from e
        self.model, _embed_dim = load_encoder(weights_path)
        self.model.eval()

    def load_random(self, seed: int = 0) -> None:
        raise NotImplementedError(
            "ECG-JEPA has no random-initialization control yet: its loader builds the "
            "transformer and loads weights in one call, and this wrapper has not verified "
            "the transformer's real constructor hyperparameters independently of a "
            "checkpoint. Wire this by inspecting models.py's encoder class directly (see "
            "docs/provenance.md's follow-up list) rather than guessing dims/depth/heads."
        )

    def preprocess(self, signal: np.ndarray) -> torch.Tensor:
        """`signal`: (samples, leads) in standard clinical lead order, as PTB-XL stores it.
        Selects the 8 leads ECG-JEPA expects (I, II, V1-V6) and resamples to 2500 samples via
        scipy.signal.resample, matching ecg_data.py. No z-scoring: ECG-JEPA's own pipeline
        doesn't normalize either."""
        x = signal[:, PTBXL_LEAD_INDICES_FOR_ECGJEPA].astype(np.float64)
        if x.shape[0] != self.INPUT_LENGTH:
            x = scipy.signal.resample(x, self.INPUT_LENGTH, axis=0)
        return torch.from_numpy(x.T.copy()).float().unsqueeze(0)  # (1, 8, 2500)

    @property
    def layer_names(self) -> List[str]:
        return ["W_P"] + [f"encoder_blocks.blocks.{i}" for i in range(12)] + ["norm"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.representation(x)
