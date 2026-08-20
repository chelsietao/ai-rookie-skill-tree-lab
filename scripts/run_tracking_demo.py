from __future__ import annotations

import argparse
import json
from pathlib import Path

from rookie_lab.database.dialects import compiled_schemas
from rookie_lab.tracking.registry import ExperimentRegistry, ExperimentRun
from rookie_lab.tracking.tensorboard_demo import write_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics/tracking.json"))
    args = parser.parse_args()
    registry = ExperimentRegistry(Path("artifacts/tmp/experiment_registry.sqlite"))
    registry.log(
        ExperimentRun(
            "demo-001", "tracking", "PyTorch", 20260820, {"profile": args.data_dir.name}, {"loss": 0.31}
        )
    )
    metrics = {
        "tensorboard": write_demo(Path("artifacts/tensorboard/demo")),
        "registry_rows": len(registry.list()),
        "compiled_rdbms_dialects": sorted(compiled_schemas()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
