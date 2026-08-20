"""Uniform metadata attached to every experiment JSON artifact."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

import psutil


def dataset_version(data_dir: Path) -> str:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    return str(manifest.get("dataset_version", manifest.get("generator", "unknown")))


def enrich_metrics(
    metrics: dict[str, Any],
    *,
    lab: str,
    profile: str,
    data_dir: Path,
    started: float,
    seed: int = 20260820,
    config_path: str | None = None,
) -> dict[str, Any]:
    process = psutil.Process()
    metrics["_run"] = {
        "lab": lab,
        "profile": profile,
        "seed": seed,
        "dataset_version": dataset_version(data_dir),
        "config_path": config_path,
        "elapsed_seconds": time.perf_counter() - started,
        "resident_memory_mb": process.memory_info().rss / 1024**2,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
    }
    return metrics
