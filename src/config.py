"""Config loading: configs/default.yaml, deep-merged with an optional extra YAML and --set overrides."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def apply_set(cfg: dict, assignment: str) -> None:
    """Apply 'a.b.c=value'; value is parsed as YAML (so 8 -> int, 0.5 -> float, null -> None)."""
    key, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    *parents, leaf = key.strip().split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(raw)


def load_config(extra: str | Path | None = None, sets: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    if extra is not None:
        cfg = deep_merge(cfg, yaml.safe_load(Path(extra).read_text()) or {})
    for s in sets or []:
        apply_set(cfg, s)
    if not cfg.get("run_name"):
        f, m = cfg["filter"], cfg["model"]
        cfg["run_name"] = f"z{m['latent_dim']}_w{m['channels'][0]}_hp{f['highpass_hz']}_{f['mode']}"
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="YAML merged on top of configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override, e.g. --set model.latent_dim=8 (repeatable)")


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
