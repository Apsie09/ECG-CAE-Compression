"""PRD / PRDN reconstruction metrics (lower is better), per window along the last axis.

PRD  = 100 * ||x - x_hat|| / ||x||
PRDN = 100 * ||x - x_hat|| / ||x - mean(x)||

Both are ratios, so a single global amplitude scale cancels out; they are still computed
in mV for clarity. After high-pass filtering windows are close to zero-mean, so PRD and
PRDN come out similar; published PRD values computed on signals with a DC/ADC offset are
not comparable.
"""

from __future__ import annotations

import numpy as np


def _ratio(num: np.ndarray, den: np.ndarray, min_den: float) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype=np.float64)
    ok = den > min_den
    out[ok] = 100.0 * num[ok] / den[ok]
    return out


def prd(x: np.ndarray, x_hat: np.ndarray, min_den: float = 1e-6) -> np.ndarray:
    return _ratio(np.linalg.norm(x - x_hat, axis=-1), np.linalg.norm(x, axis=-1), min_den)


def prdn(x: np.ndarray, x_hat: np.ndarray, min_den: float = 1e-6) -> np.ndarray:
    centred = x - x.mean(axis=-1, keepdims=True)
    return _ratio(np.linalg.norm(x - x_hat, axis=-1), np.linalg.norm(centred, axis=-1), min_den)


def summarize(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    excluded = int(np.isnan(v).sum())
    v = v[~np.isnan(v)]
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
            "mean": float(v.mean()), "n": int(v.size), "excluded": excluded}


def payload(window: int, latent_dim: int, raw_bits: int = 16, latent_bits: int = 8) -> dict[str, float]:
    """Idealised payload per non-overlapping window (no headers or metadata)."""
    raw = window * raw_bits / 8
    lat = latent_dim * latent_bits / 8
    return {"raw_bytes": raw, "latent_bytes": lat, "ratio": raw / lat,
            "raw_bits_per_sample": raw_bits, "latent_bits": latent_bits}
