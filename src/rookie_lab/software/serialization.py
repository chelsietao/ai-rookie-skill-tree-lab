"""Typed JSON and YAML round trips used by experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def dump_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(payload, sort_keys=True))
    elif path.suffix == ".json":
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    else:
        raise ValueError("supported serialization formats are JSON and YAML")


def load_payload(path: Path) -> dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        return dict(yaml.safe_load(path.read_text()))
    if path.suffix == ".json":
        return dict(json.loads(path.read_text()))
    raise ValueError("supported serialization formats are JSON and YAML")
