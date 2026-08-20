from pathlib import Path

from rookie_lab.database.dialects import compiled_schemas
from rookie_lab.tracking.registry import ExperimentRegistry, ExperimentRun


def test_registry_and_dialects(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "runs.sqlite")
    run = ExperimentRun("one", "unit", "sklearn", 7, {"depth": 2}, {"auc": 0.8})
    registry.log(run)
    assert registry.list() == [run]
    schemas = compiled_schemas()
    assert set(schemas) == {"sqlite", "mysql", "mssql"}
    assert all("CREATE TABLE" in statement for statement in schemas.values())
