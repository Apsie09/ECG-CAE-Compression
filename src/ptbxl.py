"""PTB-XL metadata, official strat_fold split and Lead II loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

NOISE_COLUMNS = ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]


def load_metadata(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    df["patient_id"] = df["patient_id"].astype(int)
    return df


def split_records(df: pd.DataFrame, folds: dict[str, list[int]],
                  max_records: dict[str, int | None] | None = None,
                  seed: int = 0) -> dict[str, pd.DataFrame]:
    """Official patient-wise split by strat_fold, optionally subsampled (deterministically)."""
    splits = {}
    for name, fold_ids in folds.items():
        part = df[df["strat_fold"].isin(fold_ids)]
        limit = (max_records or {}).get(name)
        if limit is not None and limit < len(part):
            part = part.sample(n=limit, random_state=seed).sort_index()
        splits[name] = part
    check_patient_disjoint(splits)
    return splits


def check_patient_disjoint(splits: dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(splits[a]["patient_id"]) & set(splits[b]["patient_id"])
            if shared:
                raise RuntimeError(f"{len(shared)} patients appear in both {a} and {b}")


def load_lead(root: Path, record_path: str, lead: str = "II") -> np.ndarray:
    """One record's lead as float32 in mV (PTB-XL physical units)."""
    signal, _ = wfdb.rdsamp(str(root / record_path), channel_names=[lead])
    return signal[:, 0].astype(np.float32)


def load_split(root: Path, df: pd.DataFrame, lead: str, fs: int, desc: str = "") -> np.ndarray:
    column = "filename_lr" if fs == 100 else "filename_hr"
    return np.stack([load_lead(root, p, lead) for p in tqdm(df[column], desc=desc or "load")])


def noise_flags(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """True where PTB-XL annotators noted the given signal-quality problem for the record."""
    return {c: df[c].notna().to_numpy() for c in NOISE_COLUMNS if c in df}
