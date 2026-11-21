"""Load Lead II for each split, filter whole recordings, fit the global scale, cache to .npz.

The cache key covers everything that changes the signals (data subset, lead, fs, filter),
so different filter settings get separate cache files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.config import add_config_args, load_config, resolve
from src.filtering import apply_filters
from src.ptbxl import load_metadata, load_split, noise_flags, split_records


def cache_path(cfg: dict) -> Path:
    d, f = cfg["data"], cfg["filter"]
    key = {"lead": d["lead"], "fs": d["fs"], "folds": d["folds"], "max": d["max_records"],
           "seed": cfg["seed"], "filter": f}
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    tag = f"{d['lead']}_{d['fs']}hz_hp{f['highpass_hz']}_{f['mode']}_{digest}"
    return resolve(d["processed_dir"]) / f"{tag}.npz"


def build(cfg: dict) -> Path:
    d, f = cfg["data"], cfg["filter"]
    root = resolve(d["root"])
    if not (root / "ptbxl_database.csv").exists():
        raise FileNotFoundError(f"PTB-XL not found in {root}; run `pixi run download` first")

    splits = split_records(load_metadata(root), d["folds"], d["max_records"], cfg["seed"])
    arrays: dict[str, np.ndarray] = {}
    for name, df in splits.items():
        raw = load_split(root, df, d["lead"], d["fs"], desc=f"load {name}")
        ok = np.isfinite(raw).all(axis=1)
        if not ok.all():
            print(f"{name}: dropping {(~ok).sum()} records with non-finite samples")
        df, raw = df[ok], raw[ok]
        arrays[f"{name}_signals"] = apply_filters(raw, d["fs"], f["highpass_hz"], f["order"],
                                                  f["mode"], f["lowpass_hz"])
        arrays[f"{name}_ecg_id"] = df.index.to_numpy()
        arrays[f"{name}_patient_id"] = df["patient_id"].to_numpy()
        for flag, values in noise_flags(df).items():
            arrays[f"{name}_noise_{flag}"] = values
        print(f"{name}: {len(df)} records, {df['patient_id'].nunique()} patients")

    # One global amplitude scale (mV per unit), fitted on training data only.
    # A single fixed constant is what the device would use; no per-window offset/scale
    # has to be transmitted next to the latent vector.
    arrays["scale_mv"] = np.float32(arrays["train_signals"].std())
    print(f"global scale: {float(arrays['scale_mv']):.4f} mV")

    out = cache_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"cached -> {out}")
    return out


def load_processed(cfg: dict) -> dict[str, np.ndarray]:
    path = cache_path(cfg)
    if not path.exists():
        build(cfg)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.force or not cache_path(cfg).exists():
        build(cfg)
    else:
        print(f"already cached: {cache_path(cfg)}")


if __name__ == "__main__":
    main()
