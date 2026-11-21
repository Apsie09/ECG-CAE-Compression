"""Latent-size sweep figure + table: test PRDN vs compression ratio, CAE vs PCA.

Selects evaluated runs with the given width / filter (default: w16, HP 0.5 Hz zero-phase)
and writes figures/latent_sweep.png and figures/latent_sweep.md.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.config import resolve

# Validated categorical slots 1-2 (light surface), see dataviz palette.
CAE_COLOR, PCA_COLOR = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def collect(width: int, highpass: float, mode: str) -> list[dict]:
    rows = []
    for mpath in resolve("runs").glob("*/metrics.json"):
        cfg = yaml.safe_load((mpath.parent / "config.yaml").read_text())
        f = cfg["filter"]
        if cfg["model"]["channels"][0] != width or f["highpass_hz"] != highpass or f["mode"] != mode:
            continue
        m = json.loads(mpath.read_text())
        rows.append({"z": m["latent_dim"], "ratio": m["payload"]["ratio"], "bytes": m["payload"]["latent_bytes"],
                     "enc": m["params"]["encoder"], "cae": m["per_window"]["cae_prdn"],
                     "q8": m["per_window"].get("cae_q8_prdn"),
                     "pca": m["per_window"]["pca_prdn"], "rec": m["per_record"]["cae_prdn"]["median"]})
    return sorted(rows, key=lambda r: r["z"])


def plot(rows: list[dict], title: str, path) -> None:
    rows = sorted(rows, key=lambda r: r["ratio"])          # left -> right = increasing compression
    x = [r["ratio"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for key, color, marker, ls, label in [("cae", CAE_COLOR, "o", "-", "1D CAE"),
                                          ("pca", PCA_COLOR, "s", "--", "PCA (същото k)")]:
        med = [r[key]["median"] for r in rows]
        ax.fill_between(x, [r[key]["q1"] for r in rows], [r[key]["q3"] for r in rows],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(x, med, color=color, lw=2, ls=ls, marker=marker, ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        # direct label at the right end of each line
        ax.annotate(label, (x[-1], med[-1]), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=10)
    for r in rows:  # value labels on the CAE points only
        ax.annotate(f"{r['cae']['median']:.1f}%", (r["ratio"], r["cae"]["median"]), xytext=(0, -16),
                    textcoords="offset points", ha="center", color=INK_2, fontsize=9)

    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [f"{r:.0f}:1\n(z={row['z']}, {row['bytes']:.0f} B)" for r, row in zip(x, rows)])
    ax.minorticks_off()
    ax.set_xlim(min(x) / 1.25, max(x) * 1.9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("PRDN на тестов прозорец (%)", color=INK_2)
    ax.set_xlabel("степен на компресия (16-битов прозорец спрямо int8 латентен вектор)", color=INK_2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK)
    ax.text(0.99, 0.02, "линия = медиана, лента = интерквартилен диапазон", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def table(rows: list[dict]) -> str:
    lines = ["| z | Payload (int8) | Ratio | Encoder params | CAE PRDN median (IQR), float latent | "
             "CAE PRDN median, int8 latent | CAE mean | CAE per-record median | PCA PRDN median |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r["cae"]
        q8 = f"{r['q8']['median']:.1f}%" if r["q8"] else "n/a"
        lines.append(f"| {r['z']} | {r['bytes']:.0f} B | {r['ratio']:.0f}:1 | {r['enc']:,} | "
                     f"{c['median']:.1f}% ({c['q1']:.1f}–{c['q3']:.1f}) | {q8} | {c['mean']:.1f}% | {r['rec']:.1f}% | "
                     f"{r['pca']['median']:.1f}% |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--highpass", type=float, default=0.5)
    ap.add_argument("--mode", default="zero_phase")
    args = ap.parse_args()
    rows = collect(args.width, args.highpass, args.mode)
    if len(rows) < 2:
        raise SystemExit(f"need at least 2 evaluated runs for w{args.width}, found {len(rows)}")
    out = resolve("figures")
    out.mkdir(exist_ok=True)
    title = (f"Качество на реконструкцията спрямо степента на компресия\n"
             f"PTB-XL, отвеждане II, тестов дял · ширина {args.width} · ВЧ {args.highpass} Hz")
    plot(rows, title, out / "latent_sweep.png")
    md = table(rows)
    (out / "latent_sweep.md").write_text(md + "\n")
    print(md)
    print(f"\n-> {out / 'latent_sweep.png'}")


if __name__ == "__main__":
    main()
