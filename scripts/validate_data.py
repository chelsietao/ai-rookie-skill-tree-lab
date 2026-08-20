from __future__ import annotations

import argparse
import json
from pathlib import Path

from rookie_lab.data.validate import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.data_dir), indent=2))


if __name__ == "__main__":
    main()
