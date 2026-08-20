"""A local RDBMS-backed experiment registry."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    lab: str
    framework: str
    seed: int
    config: dict[str, Any]
    metrics: dict[str, Any]


class ExperimentRegistry:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS experiment_runs (
                run_id TEXT PRIMARY KEY, lab TEXT NOT NULL, framework TEXT NOT NULL,
                seed INTEGER NOT NULL, config_json TEXT NOT NULL, metrics_json TEXT NOT NULL
                )"""
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS artifacts (
                run_id TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL,
                sha256 TEXT, PRIMARY KEY (run_id, kind, path)
                )"""
            )

    def log(self, run: ExperimentRun) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO experiment_runs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.lab,
                    run.framework,
                    run.seed,
                    json.dumps(run.config, sort_keys=True),
                    json.dumps(run.metrics, sort_keys=True),
                ),
            )

    def list(self) -> list[ExperimentRun]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                """SELECT run_id, lab, framework, seed, config_json, metrics_json
                FROM experiment_runs ORDER BY run_id"""
            ).fetchall()
        return [
            ExperimentRun(row[0], row[1], row[2], row[3], json.loads(row[4]), json.loads(row[5]))
            for row in rows
        ]

    def log_artifact(self, run_id: str, kind: str, path: str, sha256: str | None = None) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO artifacts VALUES (?, ?, ?, ?)",
                (run_id, kind, path, sha256),
            )

    def artifacts(self, run_id: str) -> list[dict[str, str | None]]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT kind, path, sha256 FROM artifacts WHERE run_id = ? ORDER BY kind, path",
                (run_id,),
            ).fetchall()
        return [{"kind": row[0], "path": row[1], "sha256": row[2]} for row in rows]
