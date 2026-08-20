from __future__ import annotations

import argparse
import json
from pathlib import Path

from rookie_lab.deep.experiment import run as run_pytorch
from rookie_lab.deep.tensorflow_lab import run as run_tensorflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("ci", "portfolio"), default="ci")
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics/deep.json"))
    args = parser.parse_args()
    metrics = {
        "pytorch": run_pytorch(args.data_dir, args.profile),
        "tensorflow": run_tensorflow(args.data_dir, args.profile),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
