import zipfile
from pathlib import Path

import pandas as pd
import pytest

from rookie_lab.data.generate import DataConfig, generate
from rookie_lab.data.validate import validate


def test_data_is_deterministic_and_valid(tmp_path: Path) -> None:
    config = DataConfig("test", 7, 80, 300, 150, 40, 40, 30, 20, 6, 24)
    first = generate(config, tmp_path / "a")
    second = generate(config, tmp_path / "b")
    assert first == second
    assert validate(tmp_path / "a")["status"] == "valid"
    assert first["rows"]["views"] == 150
    assert first["dataset_version"] == "test-7-v2"
    assert len(first["sha256"]) == 8
    assert (tmp_path / "a/customers.csv.gz").read_bytes()[4:8] == b"\x00\x00\x00\x00"
    with zipfile.ZipFile(tmp_path / "a/sequences.npz") as archive:
        assert {item.date_time for item in archive.infolist()} == {(1980, 1, 1, 0, 0, 0)}
    views = pd.read_csv(tmp_path / "a" / "product_views.csv.gz")
    assert views["view_id"].is_unique
    assert set(views["converted"]) <= {0, 1}


def test_validation_rejects_incomplete_dataset(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing generated files"):
        validate(tmp_path)
