"""Common interface for pretrained ECG model wrappers, so representation-extraction and
analysis code doesn't need to know which underlying model it's talking to.

Ported unchanged from the ecg_interp repo (src/ecg_interp/models/base.py) — this interface
was already dataset/analysis-agnostic and needed no changes to serve as the shared base for
ECGFounder, CLEF, and ECG-JEPA here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import torch


class ECGModel(ABC):
    """Wraps a pretrained ECG foundation model behind a common interface."""

    @abstractmethod
    def load(self, weights_path: str) -> None:
        """Load pretrained weights from disk."""

    @abstractmethod
    def preprocess(self, signal) -> torch.Tensor:
        """Turn a raw waveform (as loaded from PTB-XL/PTB-DB) into the model's expected
        input tensor."""

    @property
    @abstractmethod
    def layer_names(self) -> List[str]:
        """Names (as in `model.named_modules()`) of the layers to hook for representation
        extraction — early/mid/late depth, chosen per model."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model forward. Combine with `ecg_mechinterp.activation.extraction` to
        also capture intermediate activations at `layer_names` via forward hooks."""
