"""Config loading. One YAML file, dot access, CLI overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """dict that also supports attribute access, recursively."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def path(self, *keys: str) -> Path:
        """Resolve a config value that is a repo-relative path."""
        node: Any = self
        for key in keys:
            node = node[key]
        return (REPO_ROOT / str(node)).resolve()


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def resolve(rel: str | Path) -> Path:
    """Repo-relative path -> absolute."""
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()
