from __future__ import annotations

import argparse
from pathlib import Path

from rookie_lab.data.generate import DataConfig, generate
from rookie_lab.software.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("ci", "portfolio"), default="ci")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or Path("data/generated") / args.profile
    config = DataConfig.from_yaml(Path("configs/data") / f"{args.profile}.yaml")
    manifest = generate(config, output)
    get_logger("rookie_lab.data").info("generated %s", manifest["rows"])


if __name__ == "__main__":
    main()
