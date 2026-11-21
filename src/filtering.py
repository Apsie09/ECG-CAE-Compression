"""IIR filtering of whole 10-s recordings (always before windowing).

zero_phase: sosfiltfilt, offline only (uses future samples), no phase distortion.
causal:     sosfilt, what a streaming device can do; initial state set to the steady
            state for the first sample so the DC offset does not cause a start-up transient.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def highpass_sos(cutoff_hz: float, fs: float, order: int = 2) -> np.ndarray:
    return signal.butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")


def lowpass_sos(cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    return signal.butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")


def zero_phase(x: np.ndarray, sos: np.ndarray, padlen: int | None = None) -> np.ndarray:
    """sosfiltfilt along the last axis. The scipy default padding is only a few samples,
    far shorter than a 0.05-0.5 Hz high-pass settles, so pad with (almost) the whole record."""
    n = x.shape[-1]
    padlen = n - 1 if padlen is None else min(padlen, n - 1)
    return signal.sosfiltfilt(sos, x, axis=-1, padtype="odd", padlen=padlen)


def causal(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """sosfilt along the last axis, starting from the steady state for x[..., 0]."""
    zi = signal.sosfilt_zi(sos)                                 # (sections, 2)
    zi = zi.reshape(zi.shape[0], *([1] * (x.ndim - 1)), 2) * x[..., :1][None]
    y, _ = signal.sosfilt(sos, x, axis=-1, zi=zi)
    return y


def apply_filters(x: np.ndarray, fs: float, highpass_hz: float | None, order: int = 2,
                  mode: str = "zero_phase", lowpass_hz: float | None = None) -> np.ndarray:
    run = {"zero_phase": zero_phase, "causal": causal}[mode]
    y = np.asarray(x, dtype=np.float64)
    if highpass_hz:
        y = run(y, highpass_sos(highpass_hz, fs, order))
    if lowpass_hz:
        y = run(y, lowpass_sos(lowpass_hz, fs))
    return y.astype(np.float32)
