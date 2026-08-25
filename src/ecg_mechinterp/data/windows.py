"""Signal preprocessing and windowing, shared by PTB-DB (and reusable for PTB-XL 1-lead runs).

`preprocess_signal` uses the CLEF SAE convergence study's filter chain (50Hz notch -> 0.67-40Hz
bandpass -> two-stage median baseline removal), not the RQ1-4 notebook's simpler
bandpass-only chain, by deliberate choice: that study explicitly diagnosed why a plausible
alternative (a 0.05Hz "diagnostic-standard" highpass) fails — it lets baseline wander survive,
which then dominates the per-window z-score's denominator and collapses the downstream AUROC
gate to chance — and validated this chain against a >0.90 AUROC gate. The notebook's chain
was never diagnosed against that failure mode. See docs/provenance.md.

Model-specific resampling/z-scoring (which differs per model — see docs/models.md) still
happens in each model's own `preprocess()`; this module only gets a clean, windowed signal
to a common intermediate representation (native fs, filtered, per-window z-scored).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

NOTCH_HZ, NOTCH_Q = 50.0, 30.0
BAND = (0.67, 40.0)
BAND_ORDER = 4
MEDIAN_MS: Tuple[int, int] = (200, 600)  # two-stage baseline estimate


def preprocess_signal(sig: np.ndarray, fs: int) -> np.ndarray:
    """(n_samples,) raw single-lead signal -> filtered, baseline-removed signal at native fs."""
    bn, an = iirnotch(NOTCH_HZ, NOTCH_Q, fs)
    bb, ab = butter(BAND_ORDER, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="bandpass")
    x = filtfilt(bn, an, sig)
    x = filtfilt(bb, ab, x)
    k1 = int(MEDIAN_MS[0] * fs / 1000) | 1
    k2 = int(MEDIAN_MS[1] * fs / 1000) | 1
    baseline = median_filter(median_filter(x, size=k1, mode="nearest"), size=k2, mode="nearest")
    return x - baseline


def window_signal(x: np.ndarray, fs: int, window_sec: float, target_len: int,
                   max_windows: int) -> np.ndarray:
    """Filtered signal -> (n_win, target_len) float32, resampled and z-scored per window.
    Non-overlapping windows of `window_sec` seconds, capped at `max_windows`, each anti-alias
    resampled to `target_len` samples then z-scored. Raises on a flat window rather than
    silently emitting one, since a flat window is a data problem, not a modeling one."""
    n = int(round(fs * window_sec))
    n_win = min(max_windows, len(x) // n)
    if n_win == 0:
        raise ValueError(f"record shorter than one {window_sec}s window: {len(x)} samples")
    out = np.empty((n_win, target_len), dtype=np.float32)
    for i in range(n_win):
        w = x[i * n:(i + 1) * n]
        w = resample_poly(w, target_len, n)
        sd = w.std()
        if sd < 1e-8:
            raise ValueError(f"flat window {i}: std={sd:.3g}")
        out[i] = (w - w.mean()) / sd
    return out


def quality_filter(windows: np.ndarray) -> np.ndarray:
    """Boolean mask: True for windows that are neither flat nor contain non-finite values."""
    flat = windows.std(axis=1) < 1e-8
    non_finite = ~np.isfinite(windows).all(axis=1)
    return ~(flat | non_finite)
