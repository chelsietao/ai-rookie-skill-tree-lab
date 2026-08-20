from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rookie_lab.tracking.metadata import enrich_metrics
from rookie_lab.unsupervised.automl import run as run_automl
from rookie_lab.unsupervised.experiment import run as run_unsupervised


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("ci", "portfolio"), default="ci")
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics/unsupervised.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    metrics = enrich_metrics(
        {
            "unsupervised": run_unsupervised(args.data_dir, args.profile),
            "automl": run_automl(args.data_dir, args.profile),
        },
        lab="unsupervised",
        profile=args.profile,
        data_dir=args.data_dir,
        started=started,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
