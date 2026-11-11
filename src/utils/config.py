"""YAML configuration loading and project path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | Path, root: Path | None = None) -> Path:
    """Resolve a path relative to project root if not already absolute."""
    p = Path(path_like)
    if p.is_absolute():
        return p
    return (root or PROJECT_ROOT) / p


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as a dictionary."""
    full_path = resolve_path(path)
    with full_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config {full_path}, got {type(data).__name__}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_all_configs(config_dir: str | Path = "configs") -> dict[str, Any]:
    """Load base, dataset, model, and experiment configs into one dictionary."""
    cdir = resolve_path(config_dir)
    cfg: dict[str, Any] = {}
    for name in ("base.yaml", "fundus.yaml", "oct.yaml", "models.yaml", "experiments.yaml"):
        p = cdir / name
        if p.exists():
            data = load_config(p)
            if name in ("fundus.yaml", "oct.yaml"):
                key = name.replace(".yaml", "")
                cfg[key] = data.get("dataset", data)
            else:
                cfg = _deep_merge(cfg, data)
    return cfg

