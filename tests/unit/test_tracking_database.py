import json
import time
from pathlib import Path

from rookie_lab.database.dialects import compiled_schemas
from rookie_lab.tracking.metadata import dataset_version, enrich_metrics
from rookie_lab.tracking.registry import ExperimentRegistry, ExperimentRun


def test_registry_and_dialects(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.sqlite")
    run = ExperimentRun("one", "unit", "sklearn", 7, {"depth": 2}, {"auc": 0.8})
    registry.log(run)
    assert registry.list() == [run]
    schemas = compiled_schemas()
    assert set(schemas) == {"sqlite", "mysql", "mssql"}
    assert all("CREATE TABLE" in statement for statement in schemas.values())
    registry.log_artifact("one", "model", "artifacts/model.joblib", "abc")
    assert registry.artifacts("one") == [
        {"kind": "model", "path": "artifacts/model.joblib", "sha256": "abc"}
    ]


def test_uniform_run_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(json.dumps({"dataset_version": "unit-v2"}))
    payload = enrich_metrics(
        {"score": 0.5}, lab="unit", profile="test", data_dir=data_dir, started=time.perf_counter()
    )
    assert dataset_version(data_dir) == "unit-v2"
    assert payload["_run"]["seed"] == 20260820
    assert payload["_run"]["resident_memory_mb"] > 0


def test_sql_schemas_cover_product_views() -> None:
    root = Path(__file__).resolve().parents[2]
    for dialect in ("sqlite", "mysql", "mssql"):
        assert "product_views" in (root / "sql" / dialect / "schema.sql").read_text()
