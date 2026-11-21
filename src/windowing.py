"""Fixed-length sliding windows over already-filtered recordings."""

from __future__ import annotations

import numpy as np


def make_windows(signals: np.ndarray, size: int, stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """signals: [n_records, length] -> windows [N, size], record index [N], start sample [N].

    Samples after the last full window are dropped (1000 samples, size 256, stride 128
    -> 6 windows per record; stride 256 -> 3 windows).
    """
    n_rec, length = signals.shape
    if length < size:
        raise ValueError(f"records ({length}) shorter than window ({size})")
    starts = np.arange(0, length - size + 1, stride)
    view = np.lib.stride_tricks.sliding_window_view(signals, size, axis=1)[:, starts]
    windows = np.ascontiguousarray(view.reshape(-1, size))
    rec_idx = np.repeat(np.arange(n_rec), len(starts))
    start = np.tile(starts, n_rec)
    return windows, rec_idx, start
