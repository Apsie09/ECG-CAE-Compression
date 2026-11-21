"""Post-training int8 quantization of the latent vector only (the model stays float32).

Per latent dimension d, an affine map fitted on TRAINING latents:
    q_d = clip(round((z_d - lo_d) / step_d), 0, 255) - 128        (int8, sent over the link)
    z_d ~= (q_d + 128) * step_d + lo_d                             (receiver side)
lo/hi are robust percentiles (not min/max) so a few outliers don't waste resolution;
values outside are clipped. lo and step are fixed constants known to both encoder and
decoder, so they are NOT part of the per-window payload.
"""

from __future__ import annotations

import numpy as np

LEVELS = 255


def fit(z_train: np.ndarray, clip_pct: float = 0.1) -> dict[str, np.ndarray]:
    lo = np.percentile(z_train, clip_pct, axis=0)
    hi = np.percentile(z_train, 100 - clip_pct, axis=0)
    step = np.maximum((hi - lo) / LEVELS, 1e-8)
    return {"lo": lo.astype(np.float32), "step": step.astype(np.float32)}


def quantize(z: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    q = np.clip(np.round((z - params["lo"]) / params["step"]), 0, LEVELS) - 128
    return q.astype(np.int8)


def dequantize(q: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    return ((q.astype(np.float32) + 128) * params["step"] + params["lo"]).astype(np.float32)


def clip_rate(z: np.ndarray, params: dict[str, np.ndarray]) -> float:
    """Fraction of latent values outside the fitted range (saturated by quantization)."""
    hi = params["lo"] + LEVELS * params["step"]
    return float(((z < params["lo"]) | (z > hi)).mean())
