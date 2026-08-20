from pathlib import Path

from rookie_lab.data.generate import DataConfig, generate
from rookie_lab.data.validate import validate


def test_data_is_deterministic_and_valid(tmp_path: Path) -> None:
    config = DataConfig("test", 7, 80, 300, 40, 40, 30, 20, 6, 24)
    first = generate(config, tmp_path / "a")
    second = generate(config, tmp_path / "b")
    assert first == second
    assert validate(tmp_path / "a")["status"] == "valid"
