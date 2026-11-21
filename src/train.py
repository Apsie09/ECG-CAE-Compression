"""Train one CAE on the training folds, early-stop on validation MSE.

Outputs runs/<run_name>/: model.pt (weights + config + scale), config.yaml, history.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from src.cae import build_model, count_params
from src.config import add_config_args, load_config, resolve
from src.metrics import prdn
from src.preprocess import load_processed
from src.windowing import make_windows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def windows_tensor(signals: np.ndarray, size: int, stride: int, scale: float, device) -> torch.Tensor:
    w, _, _ = make_windows(signals, size, stride)
    return torch.from_numpy(w / scale).float().unsqueeze(1).to(device)     # [N, 1, size]


def recon_figure(x: torch.Tensor, x_hat: torch.Tensor, scale: float, fs: int, scores) -> plt.Figure:
    """Original vs reconstruction (mV) for a handful of windows, for TensorBoard."""
    x = x.squeeze(1).cpu().numpy() * scale
    x_hat = x_hat.squeeze(1).cpu().numpy() * scale
    t = np.arange(x.shape[1]) / fs
    fig, axes = plt.subplots(1, len(x), figsize=(4 * len(x), 2.6), squeeze=False)
    for ax, a, b, s in zip(axes[0], x, x_hat, scores):
        ax.plot(t, a, "k", lw=1.1, label="original")
        ax.plot(t, b, "C3", lw=1.0, label="CAE")
        ax.set_title(f"PRDN {s:.1f}%", fontsize=9)
        ax.set_xlabel("s")
    axes[0, 0].set_ylabel("mV")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return fig


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    model.eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(cfg["seed"])
    device = pick_device(cfg["train"]["device"])
    run_dir = resolve("runs") / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(run_dir / "tb", ignore_errors=True)       # fresh curves when a run is repeated

    data = load_processed(cfg)
    scale = float(data["scale_mv"])
    w = cfg["window"]
    x_train = windows_tensor(data["train_signals"], w["size"], w["train_stride"], scale, device)
    x_val = windows_tensor(data["val_signals"], w["size"], w["eval_stride"], scale, device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU (no CUDA GPU in use!)"
    print(f"train windows {len(x_train)}, val windows {len(x_val)}, device {device}: {gpu}")

    model = build_model(cfg).to(device)
    print(f"params: encoder {count_params(model.encoder)}, decoder {count_params(model.decoder)}")
    t = cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    sched_cfg = t.get("scheduler")
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=sched_cfg["factor"], patience=sched_cfg["patience"],
                                                        min_lr=sched_cfg["min_lr"]) if sched_cfg else None)
    loss_fn = nn.MSELoss()
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])

    tb = SummaryWriter(log_dir=str(run_dir / "tb"), purge_step=0)
    tb.add_text("config", f"```\n{yaml.safe_dump(cfg, sort_keys=False)}\n```")
    # Fixed validation windows (spread over the set) to watch reconstructions improve.
    show_idx = torch.linspace(0, len(x_val) - 1, 4).long()
    n_batches = -(-len(x_train) // t["batch_size"])
    log_every = max(1, n_batches // 50)

    best, best_epoch, bad, history = float("inf"), 0, 0, []
    epochs = trange(1, t["epochs"] + 1, desc=cfg["run_name"], unit="epoch", position=0, dynamic_ncols=True)
    for epoch in epochs:
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(x_train), device=device, generator=gen)
        total, seen = 0.0, 0
        batches = tqdm(range(0, len(perm), t["batch_size"]), total=n_batches, desc=f"  epoch {epoch}",
                       unit="batch", position=1, leave=False, dynamic_ncols=True)
        for step, i in enumerate(batches):
            xb = x_train[perm[i:i + t["batch_size"]]]
            loss = loss_fn(model(xb), xb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            seen += len(xb)
            if step % log_every == 0:
                batches.set_postfix(mse=f"{total / seen:.5f}")
                tb.add_scalar("batch/train_mse", loss.item(), (epoch - 1) * n_batches + step)
        train_mse = total / seen

        recon = predict(model, x_val)
        val_mse = loss_fn(recon, x_val).item()
        val_prdn_all = prdn(x_val.squeeze(1).cpu().numpy(), recon.squeeze(1).cpu().numpy())
        val_prdn = float(np.nanmedian(val_prdn_all))
        seconds = time.time() - t0
        lr = opt.param_groups[0]["lr"]                   # lr used during this epoch
        if sched:
            sched.step(val_mse)
        history.append({"epoch": epoch, "train_mse": train_mse, "val_mse": val_mse,
                        "val_prdn_median": val_prdn, "lr": lr, "seconds": seconds})

        improved = val_mse < best
        if improved:
            best, best_epoch, bad = val_mse, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg, "scale_mv": scale,
                        "epoch": epoch}, run_dir / "model.pt")
        else:
            bad += 1

        tb.add_scalars("mse", {"train": train_mse, "val": val_mse}, epoch)
        tb.add_scalar("val/prdn_median_pct", val_prdn, epoch)
        tb.add_histogram("val/prdn_pct", val_prdn_all[np.isfinite(val_prdn_all)].clip(0, 200), epoch)
        tb.add_scalar("epoch_seconds", seconds, epoch)
        tb.add_scalar("lr", lr, epoch)
        tb.add_figure("val/reconstructions", recon_figure(x_val[show_idx], recon[show_idx], scale,
                                                          cfg["data"]["fs"], val_prdn_all[show_idx.numpy()]), epoch)
        tb.flush()

        epochs.set_postfix(train=f"{train_mse:.5f}", val=f"{val_mse:.5f}", prdn=f"{val_prdn:.2f}%",
                           lr=f"{lr:.0e}", best=f"e{best_epoch}", patience=f"{bad}/{t['patience']}")
        tqdm.write(f"epoch {epoch:3d} | train {train_mse:.5f} | val {val_mse:.5f} | "
                   f"val PRDN median {val_prdn:6.2f}% | lr {lr:.0e} | {seconds:5.1f}s "
                   f"{'| * saved' if improved else ''}")
        if bad >= t["patience"]:
            tqdm.write(f"early stop: no val improvement for {bad} epochs")
            break
    epochs.close()
    tb.close()

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"best val MSE {best:.5f} at epoch {best_epoch} -> {run_dir}")


if __name__ == "__main__":
    main()
