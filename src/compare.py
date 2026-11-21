"""Collect runs/*/metrics.json into runs/summary.csv (one row per run) for the report."""

from __future__ import annotations

import json

import pandas as pd

from src.config import resolve


def main() -> None:
    rows = []
    for path in sorted(resolve("runs").glob("*/metrics.json")):
        m = json.loads(path.read_text())
        rows.append({
            "run": m["run"], "latent_dim": m["latent_dim"],
            "highpass_hz": m["filter"]["highpass_hz"], "filter_mode": m["filter"]["mode"],
            "encoder_params": m["params"]["encoder"], "decoder_params": m["params"]["decoder"],
            "payload_ratio": m["payload"]["ratio"],
            **{f"cae_prdn_{s}": m["per_window"]["cae_prdn"][s] for s in ("median", "q1", "q3", "mean")},
            "cae_q8_prdn_median": m["per_window"].get("cae_q8_prdn", {}).get("median"),
            "cae_prd_median": m["per_window"]["cae_prd"]["median"],
            "cae_prdn_record_median": m["per_record"]["cae_prdn"]["median"],
            "pca_prdn_median": m["per_window"]["pca_prdn"]["median"],
            "test_windows": m["test"]["windows"],
        })
    if not rows:
        print("no evaluated runs found (run `pixi run evaluate --all` first)")
        return
    df = pd.DataFrame(rows).sort_values(["highpass_hz", "filter_mode", "latent_dim"])
    out = resolve("runs") / "summary.csv"
    df.to_csv(out, index=False, float_format="%.3f")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
