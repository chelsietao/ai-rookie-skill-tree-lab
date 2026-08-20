import re
from pathlib import Path


def test_no_secret_patterns_in_tracked_source() -> None:
    patterns = [re.compile(r"sk-[A-Za-z0-9_-]{20,}"), re.compile(r"AKIA[0-9A-Z]{16}")]
    excluded = {".git", ".venv", "data/generated", "artifacts"}
    for path in Path(".").rglob("*"):
        relative = str(path)
        if not path.is_file() or any(value in relative for value in excluded):
            continue
        if path.suffix.lower() not in {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".sql"}:
            continue
        text = path.read_text(errors="ignore")
        assert not any(pattern.search(text) for pattern in patterns), relative
