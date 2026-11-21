"""Evaluate trained runs on the held-out test fold (10), next to a PCA baseline of the same size.

Test windows use eval_stride (non-overlapping by default). Metrics are reported per window
and per recording (median over the recording's windows), since windows from one recording
are correlated. Writes metrics.json, per_window.csv and figures/ into each run directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src import latent_quant
from src.cae import build_model, count_params
from src.config import resolve
from src.metrics import payload, prd, prdn, summarize
from src.preprocess import load_processed
from src.train import pick_device, predict
from src.windowing import make_windows


def pca_baseline(train_windows: np.ndarray, test_windows: np.ndarray, k: int, n_fit: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_windows), size=min(n_fit, len(train_windows)), replace=False)
    pca = PCA(n_components=k, random_state=seed).fit(train_windows[idx])
    return pca.inverse_transform(pca.transform(test_windows)), float(pca.explained_variance_ratio_.sum())


@torch.no_grad()
def encode(model, x: torch.Tensor, batch: int = 8192) -> np.ndarray:
    model.eval()
    return torch.cat([model.encoder(x[i:i + batch]) for i in range(0, len(x), batch)]).cpu().numpy()


@torch.no_grad()
def decode(model, z: np.ndarray, device, batch: int = 8192) -> np.ndarray:
    model.eval()
    zt = torch.from_numpy(z).float().to(device)
    return torch.cat([model.decoder(zt[i:i + batch]) for i in range(0, len(zt), batch)]).squeeze(1).cpu().numpy()


def per_record(values: np.ndarray, rec_idx: np.ndarray) -> np.ndarray:
    return pd.Series(values).groupby(rec_idx).median().to_numpy()


EN = {"best": "best", "median": "median", "worst": "worst", "original": "original", "time": "time (s)"}


def plot_examples(x, x_cae, x_pca, scores, fs, n, path: Path, title: str, text: dict = EN) -> None:
    order = np.argsort(np.nan_to_num(scores, nan=np.inf))
    order = order[: np.isfinite(scores).sum()]
    picks = {text["best"]: order[:n], text["median"]: order[len(order) // 2 - n // 2:][:n], text["worst"]: order[-n:][::-1]}
    t = np.arange(x.shape[1]) / fs
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 7.5), sharex=True, squeeze=False)
    for row, (label, ids) in enumerate(picks.items()):
        for col, i in enumerate(ids):
            ax = axes[row, col]
            ax.plot(t, x[i], "k", lw=1.2, label=text["original"])
            ax.plot(t, x_cae[i], "C3", lw=1.0, label="CAE")
            ax.plot(t, x_pca[i], "C0", lw=0.8, ls="--", alpha=0.8, label="PCA")
            ax.set_title(f"{label}: PRDN {scores[i]:.1f}%", fontsize=9)
            if col == 0:
                ax.set_ylabel("mV")
    for ax in axes[-1]:
        ax.set_xlabel(text["time"])
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distribution(cae, pca, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    hi = np.nanpercentile(np.concatenate([cae, pca]), 99)
    bins = np.linspace(0, hi, 60)
    ax.hist(cae[np.isfinite(cae)], bins, alpha=0.6, label=f"CAE (median {np.nanmedian(cae):.1f}%)")
    ax.hist(pca[np.isfinite(pca)], bins, alpha=0.6, label=f"PCA (median {np.nanmedian(pca):.1f}%)")
    ax.set_xlabel("PRDN (%) per test window")
    ax.set_ylabel("windows")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(run_dir: Path, path: Path) -> None:
    hist_path = run_dir / "history.csv"
    if not hist_path.exists():
        return
    h = pd.read_csv(hist_path)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(h["epoch"], h["train_mse"], label="train MSE")
    ax.plot(h["epoch"], h["val_mse"], label="val MSE")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(run_dir: Path) -> dict:
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    cfg, scale = ckpt["config"], ckpt["scale_mv"]
    device = pick_device(cfg["train"]["device"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    data = load_processed(cfg)
    w, e, k = cfg["window"], cfg["eval"], cfg["model"]["latent_dim"]
    x, rec_idx, start = make_windows(data["test_signals"], w["size"], w["eval_stride"])   # mV
    x_in = torch.from_numpy(x / scale).float().unsqueeze(1).to(device)
    x_cae = predict(model, x_in).squeeze(1).cpu().numpy() * scale

    train_w, _, _ = make_windows(data["train_signals"], w["size"], w["train_stride"])
    x_pca, pca_var = pca_baseline(train_w, x, k, e["pca_fit_windows"], cfg["seed"])

    # int8 latent: fit per-dimension ranges on TRAIN latents, quantize test latents, decode.
    z_train = encode(model, torch.from_numpy(train_w / scale).float().unsqueeze(1).to(device))
    qparams = latent_quant.fit(z_train)
    z_test = encode(model, x_in)
    z_q = latent_quant.dequantize(latent_quant.quantize(z_test, qparams), qparams)
    x_q8 = decode(model, z_q, device) * scale
    np.savez(run_dir / "latent_quant.npz", **qparams)

    md = e["min_denominator"]
    scores = {"cae_prd": prd(x, x_cae, md), "cae_prdn": prdn(x, x_cae, md),
              "cae_q8_prd": prd(x, x_q8, md), "cae_q8_prdn": prdn(x, x_q8, md),
              "pca_prd": prd(x, x_pca, md), "pca_prdn": prdn(x, x_pca, md)}

    noise = {key[len("test_noise_"):]: data[key][rec_idx] for key in data if key.startswith("test_noise_")}
    noise_breakdown = {}
    for flag, mask in noise.items():
        noise_breakdown[flag] = {
            "windows_flagged": int(mask.sum()),
            "cae_prdn_median_flagged": float(np.nanmedian(scores["cae_prdn"][mask])) if mask.any() else None,
            "cae_prdn_median_clean": float(np.nanmedian(scores["cae_prdn"][~mask])),
        }

    result = {
        "run": run_dir.name,
        "latent_dim": k,
        "filter": cfg["filter"],
        "best_epoch": ckpt["epoch"],
        "params": {"encoder": count_params(model.encoder), "decoder": count_params(model.decoder)},
        "test": {"records": int(len(data["test_ecg_id"])), "windows": int(len(x)),
                 "eval_stride": w["eval_stride"]},
        "per_window": {name: summarize(v) for name, v in scores.items()},
        "per_record": {name: summarize(per_record(v, rec_idx)) for name, v in scores.items()},
        "pca_explained_variance": pca_var,
        "latent_int8": {"test_clip_rate": latent_quant.clip_rate(z_test, qparams),
                        "mean_step": float(qparams["step"].mean()),
                        "latent_rmse": float(np.sqrt(((z_q - z_test) ** 2).mean()))},
        "payload": payload(w["size"], k),
        "payload_float32_latent": payload(w["size"], k, latent_bits=32),
        "noise_breakdown": noise_breakdown,
    }
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))

    with open(run_dir / "per_window.csv", "w", newline="") as f:
        cols = ["ecg_id", "patient_id", "start", *scores, *noise]
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(len(x)):
            r = rec_idx[i]
            writer.writerow([data["test_ecg_id"][r], data["test_patient_id"][r], start[i],
                             *(f"{scores[s][i]:.4f}" for s in scores), *(int(noise[n][i]) for n in noise)])

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    title = f"z={k}, HP {cfg['filter']['highpass_hz']} Hz ({cfg['filter']['mode']})"
    plot_examples(x, x_cae, x_pca, scores["cae_prdn"], cfg["data"]["fs"], e["n_examples"],
                  fig_dir / "examples.png", title)
    plot_distribution(scores["cae_prdn"], scores["pca_prdn"], fig_dir / "prdn_hist.png", title)
    plot_history(run_dir, fig_dir / "history.png")

    pw = result["per_window"]
    print(f"{run_dir.name}: CAE PRDN median {pw['cae_prdn']['median']:.2f}% "
          f"(IQR {pw['cae_prdn']['q1']:.2f}-{pw['cae_prdn']['q3']:.2f}), "
          f"int8 latent {pw['cae_q8_prdn']['median']:.2f}%, "
          f"PCA PRDN median {pw['pca_prdn']['median']:.2f}%, payload ratio {result['payload']['ratio']:.0f}:1")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", default=[], help="run directory (repeatable)")
    ap.add_argument("--all", action="store_true", help="evaluate every runs/*/model.pt")
    args = ap.parse_args()
    runs = [resolve(r) for r in args.run]
    if args.all:
        runs += sorted(p.parent for p in resolve("runs").glob("*/model.pt"))
    if not runs:
        ap.error("give --run DIR or --all")
    for run in dict.fromkeys(runs):
        evaluate(run)


if __name__ == "__main__":
    main()
