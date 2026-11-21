"""Report figures that combine several runs (w16 latent sweep + final causal model):
  figures/recon_by_z.png        the same test windows reconstructed with z = 8 / 16 / 32
  figures/training_curves.png   validation PRDN per epoch for the sweep runs
  figures/final_examples.png    best / median / worst windows of the final (causal) model
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.cae import build_model
from src.config import resolve
from src.metrics import prdn
from src.preprocess import load_processed
from src.windowing import make_windows

SWEEP = {8: "z8_w16_hp0.5_zero_phase", 16: "z16_w16_hp0.5_zero_phase", 32: "z32_w16_hp0.5_zero_phase"}
FINAL = "z32_w16_hp0.5_causal"
# ordinal blue ramp (dataviz palette steps 250 / 450 / 650): light = most compressed
Z_COLOR = {8: "#86b6ef", 16: "#2a78d6", 32: "#104281"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def load_model(run: str):
    ck = torch.load(resolve("runs") / run / "model.pt", map_location="cpu", weights_only=False)
    model = build_model(ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def recon_by_z(out) -> None:
    _, ck = load_model(SWEEP[32])
    data = load_processed(ck["config"])
    x, _, _ = make_windows(data["test_signals"], 256, 256)
    rec = {}
    for z, run in SWEEP.items():
        model, ck = load_model(run)
        with torch.no_grad():
            rec[z] = model(torch.from_numpy(x / ck["scale_mv"]).float().unsqueeze(1)).squeeze(1).numpy() * ck["scale_mv"]
    score = prdn(x, rec[32])
    order = np.argsort(np.nan_to_num(score, nan=np.inf))
    n = np.isfinite(score).sum()
    picks = [("добър (10-и перцентил)", order[int(0.10 * n)]),
             ("типичен (медиана)", order[int(0.50 * n)]),
             ("труден (90-и перцентил)", order[int(0.90 * n)])]
    t = np.arange(256) / 100
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.6), sharex=True, facecolor=SURFACE)
    for ax, (label, i) in zip(axes, picks):
        style(ax)
        ax.plot(t, x[i], color=INK, lw=1.5, label="оригинал")
        for z in SWEEP:
            ax.plot(t, rec[z][i], color=Z_COLOR[z], lw=1.2, label=f"z = {z}")
        scores = ", ".join(f"z={z}: {prdn(x[i:i+1], rec[z][i:i+1])[0]:.1f}%" for z in SWEEP)
        ax.set_title(f"Прозорец – {label};  PRDN {scores}", fontsize=9, color=INK, loc="left")
        ax.set_ylabel("mV", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False, ncol=4, loc="upper center")
    axes[-1].set_xlabel("време (s)", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "recon_by_z.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def training_curves(out) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    style(ax)
    for z, run in SWEEP.items():
        h = pd.read_csv(resolve("runs") / run / "history.csv")
        ax.plot(h["epoch"], h["val_prdn_median"], color=Z_COLOR[z], lw=2)
        ax.annotate(f"z = {z}", (h["epoch"].iloc[-1], h["val_prdn_median"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xlabel("епоха", color=INK_2, fontsize=9)
    ax.set_ylabel("медиана на PRDN, валидация (%)", color=INK_2, fontsize=9)
    ax.set_xlim(0, 170)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def final_examples(out) -> None:
    """Best / median / worst test windows of the final causal model, CAE vs PCA (Bulgarian labels)."""
    from src.evaluate import pca_baseline, plot_examples

    model, ck = load_model(FINAL)
    cfg = ck["config"]
    data = load_processed(cfg)
    x, _, _ = make_windows(data["test_signals"], 256, 256)
    with torch.no_grad():
        x_cae = model(torch.from_numpy(x / ck["scale_mv"]).float().unsqueeze(1)).squeeze(1).numpy() * ck["scale_mv"]
    train_w, _, _ = make_windows(data["train_signals"], 256, 128)
    x_pca, _ = pca_baseline(train_w, x, cfg["model"]["latent_dim"], cfg["eval"]["pca_fit_windows"], cfg["seed"])
    text = {"best": "най-добър", "median": "медиана", "worst": "най-лош", "original": "оригинал", "time": "време (s)"}
    plot_examples(x, x_cae, x_pca, prdn(x, x_cae), 100, 3, out / "final_examples.png",
                  "Финален модел: z = 32, ширина 16, каузален ВЧ филтър 0.5 Hz", text)


def cae_thumbnails(out) -> None:
    """Small input / reconstruction traces used inside the CAE architecture diagram."""
    model, ck = load_model(FINAL)
    data = load_processed(ck["config"])
    x, _, _ = make_windows(data["test_signals"], 256, 256)
    with torch.no_grad():
        xh = model(torch.from_numpy(x / ck["scale_mv"]).float().unsqueeze(1)).squeeze(1).numpy() * ck["scale_mv"]
    score = prdn(x, xh)
    order = np.argsort(np.nan_to_num(score, nan=np.inf))
    i = order[int(0.25 * np.isfinite(score).sum())]          # a clean, typical window
    for name, traces in [("cae_in", [(x[i], INK, 1.4)]),
                         ("cae_out", [(x[i], "#b8b7b2", 1.2), (xh[i], "#d03b3b", 1.4)])]:
        fig, ax = plt.subplots(figsize=(2.4, 1.2), facecolor="white")
        for y, c, lw in traces:
            ax.plot(y, color=c, lw=lw)
        ax.axis("off")
        ax.margins(x=0.01, y=0.08)
        fig.savefig(out / f"{name}.png", dpi=250, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main() -> None:
    out = resolve("figures")
    out.mkdir(exist_ok=True)
    recon_by_z(out)
    training_curves(out)
    final_examples(out)
    cae_thumbnails(out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
