import numpy as np
import pytest
import torch

from src.cae import CAE
from src.config import load_config
from src.filtering import apply_filters
from src.metrics import payload, prd, prdn, summarize
from src.windowing import make_windows


def test_windows_per_record():
    sig = np.arange(2 * 1000, dtype=np.float32).reshape(2, 1000)
    w, rec, start = make_windows(sig, 256, 128)
    assert w.shape == (12, 256)                      # 6 per record
    assert list(start[:6]) == [0, 128, 256, 384, 512, 640]
    assert (rec == np.repeat([0, 1], 6)).all()
    assert np.array_equal(w[7], sig[1, 128:384])
    assert make_windows(sig, 256, 256)[0].shape == (6, 256)


def test_metrics():
    x = np.sin(np.linspace(0, 6 * np.pi, 256))[None]
    assert prd(x, x)[0] == pytest.approx(0.0)
    assert prdn(x, 0.9 * x)[0] == pytest.approx(10.0, rel=1e-2)
    # PRDN ignores DC offset in the denominator, PRD does not
    assert prd(x + 5, 0.9 * x + 5)[0] < prdn(x + 5, 0.9 * x + 5)[0]
    assert np.isnan(prdn(np.ones((1, 256)), np.ones((1, 256)))[0])   # constant window excluded
    s = summarize(np.array([1.0, 2.0, 3.0, np.nan]))
    assert s["median"] == 2.0 and s["excluded"] == 1
    assert payload(256, 16)["ratio"] == 32


@pytest.mark.parametrize("mode", ["zero_phase", "causal"])
def test_highpass_removes_drift(mode):
    fs, t = 100, np.arange(1000) / 100
    ecg_like = np.sin(2 * np.pi * 1.2 * t)
    drifted = ecg_like + 2.0 + 0.5 * np.sin(2 * np.pi * 0.05 * t)
    y = apply_filters(drifted[None], fs, 0.5, 2, mode)[0]
    assert abs(y[200:].mean()) < 0.1                 # DC and slow drift gone after settling
    if mode == "zero_phase":
        assert np.corrcoef(y, ecg_like)[0, 1] > 0.95


@pytest.mark.parametrize("z", [8, 16, 32])
def test_cae_shapes(z):
    model = CAE(latent_dim=z)
    x = torch.randn(4, 1, 256)
    assert model.encoder(x).shape == (4, z)
    assert model(x).shape == x.shape


def test_latent_int8_roundtrip():
    from src import latent_quant
    rng = np.random.default_rng(0)
    z = rng.normal(loc=[0.0, 5.0, -3.0], scale=[1.0, 0.1, 10.0], size=(5000, 3)).astype(np.float32)
    p = latent_quant.fit(z)
    q = latent_quant.quantize(z, p)
    assert q.dtype == np.int8
    err = np.abs(latent_quant.dequantize(q, p) - z)
    inside = latent_quant.clip_rate(z, p)
    assert inside < 0.01                              # ~0.2 % clipped by the 0.1/99.9 pct range
    ok = (z >= p["lo"]) & (z <= p["lo"] + 255 * p["step"])
    assert (err[ok] <= p["step"].repeat(len(z)).reshape(3, -1).T[ok] / 2 + 1e-6).all()


def test_config_overrides():
    cfg = load_config(sets=["model.latent_dim=8", "filter.highpass_hz=0.05"])
    assert cfg["model"]["latent_dim"] == 8
    assert cfg["run_name"] == "z8_w8_hp0.05_zero_phase"
    wide = load_config(sets=["model.channels=[16, 32, 64]"])
    assert wide["model"]["channels"] == [16, 32, 64] and wide["run_name"].startswith("z16_w16_")
    assert isinstance(cfg["train"]["lr"], float)
