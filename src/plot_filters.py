"""Figures for the filtering section of the report:
  figures/highpass_response.png  magnitude/phase of the 0.05 and 0.5 Hz high-pass filters
  figures/filter_example.png     one raw Lead II record vs zero-phase and causal filtering
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from src.config import add_config_args, load_config, resolve
from src.filtering import apply_filters, highpass_sos
from src.ptbxl import load_lead, load_metadata


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--ecg-id", type=int, default=None, help="record to plot (default: one with baseline drift)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    d, order = cfg["data"], cfg["filter"]["order"]
    fs, cutoffs = d["fs"], (0.05, 0.5)
    out = resolve("figures")
    out.mkdir(exist_ok=True)

    fig, (ax_m, ax_p) = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True)
    for fc in cutoffs:
        f, h = signal.sosfreqz(highpass_sos(fc, fs, order), worN=np.logspace(-3, np.log10(fs / 2), 2000), fs=fs)
        ax_m.semilogx(f, 20 * np.log10(np.abs(h) + 1e-12), label=f"{fc} Hz")
        ax_p.semilogx(f, np.degrees(np.unwrap(np.angle(h))), label=f"{fc} Hz")
    ax_m.set_ylim(-60, 5)
    ax_m.set_ylabel("амплитуда (dB)")
    ax_p.set_ylabel("фаза (°), каузален")
    ax_p.set_xlabel("честота (Hz)")
    ax_m.set_title(f"Високочестотен филтър на Бътъруърт, ред {order}, fs = {fs} Hz")
    ax_m.legend()
    fig.tight_layout()
    fig.savefig(out / "highpass_response.png", dpi=150)
    plt.close(fig)

    root = resolve(d["root"])
    meta = load_metadata(root)
    if args.ecg_id is None:
        drift = meta[meta["baseline_drift"].notna() & meta["strat_fold"].isin(d["folds"]["train"])]
        ecg_id = int(drift.index[0] if len(drift) else meta.index[0])
    else:
        ecg_id = args.ecg_id
    raw = load_lead(root, meta.loc[ecg_id, "filename_lr" if fs == 100 else "filename_hr"], d["lead"])
    t = np.arange(raw.size) / fs

    fig, axes = plt.subplots(len(cutoffs) + 1, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, raw, "k", lw=0.8)
    axes[0].set_title(f"Необработен сигнал, отвеждане II, ecg_id {ecg_id}")
    for ax, fc in zip(axes[1:], cutoffs):
        ax.plot(t, apply_filters(raw, fs, fc, order, "zero_phase"), lw=0.9, label="нулева фаза (sosfiltfilt)")
        ax.plot(t, apply_filters(raw, fs, fc, order, "causal"), lw=0.9, alpha=0.8, label="каузален (sosfilt)")
        ax.set_title(f"високочестотен филтър {fc} Hz")
        ax.legend(fontsize=8, loc="upper right")
    for ax in axes:
        ax.set_ylabel("mV")
    axes[-1].set_xlabel("време (s)")
    fig.tight_layout()
    fig.savefig(out / "filter_example.png", dpi=150)
    plt.close(fig)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
