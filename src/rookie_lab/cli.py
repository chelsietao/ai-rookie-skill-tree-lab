"""Command-line entry point."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rookie_lab.data.generate import DataConfig, generate
from rookie_lab.data.validate import validate


def main() -> None:
    parser = argparse.ArgumentParser(prog="rookie-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--profile", choices=("ci", "portfolio"), default="ci")
    reproduce = subparsers.add_parser("reproduce")
    reproduce.add_argument("--profile", choices=("ci", "portfolio"), default="ci")
    subparsers.add_parser("serve")
    args = parser.parse_args()
    if args.command == "prepare-data":
        config = DataConfig.from_yaml(Path("configs/data") / f"{args.profile}.yaml")
        output = Path("data/generated") / args.profile
        generate(config, output)
        print(validate(output))
    elif args.command == "reproduce":
        subprocess.run(
            [sys.executable, "scripts/reproduce_all.py", "--profile", args.profile], check=True
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "uvicorn", "rookie_lab.serving.api:app", "--host", "127.0.0.1"],
            check=True,
        )
