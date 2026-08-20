from pathlib import Path

import pytest

from rookie_lab.software.serialization import dump_payload, load_payload


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_serialization_round_trip(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"payload{suffix}"
    payload = {"seed": 7, "metric": 0.9}
    dump_payload(payload, path)
    assert load_payload(path) == payload
