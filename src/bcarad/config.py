"""Load a YAML config (optionally inheriting another one) and resolve its paths against the repository root."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path):
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("inherit", None)
    return _merge(_read(path.parent / parent), cfg) if parent else cfg


def load_config(path):
    cfg = _read(path)
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = (REPO_ROOT / value).resolve()
    cfg["out"] = cfg["paths"]["out_dir"] / cfg["run_name"]
    cfg["out"].mkdir(parents=True, exist_ok=True)
    return cfg


def out_path(cfg, *parts):
    """Path inside outputs/<run_name>/, creating parent folders."""
    p = cfg["out"].joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
