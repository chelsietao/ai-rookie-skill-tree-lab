from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def call(*arguments: str) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("ci", "portfolio"), default="ci")
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts/tmp/matplotlib").resolve()))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    data_dir = Path("data/generated") / args.profile
    metrics = Path("artifacts/metrics") if args.profile == "portfolio" else Path("artifacts/metrics/ci")
    call(sys.executable, "scripts/prepare_data.py", "--profile", args.profile, "--output", str(data_dir))
    call(sys.executable, "scripts/validate_data.py", "--data-dir", str(data_dir))
    for script, name in (
        ("run_classical.py", "classical"),
        ("run_unsupervised.py", "unsupervised"),
        ("run_deep.py", "deep"),
        ("run_vision.py", "vision"),
        ("run_nlp.py", "nlp"),
        ("run_graph.py", "graph"),
    ):
        call(
            sys.executable,
            f"scripts/{script}",
            "--data-dir",
            str(data_dir),
            "--profile",
            args.profile,
            "--output",
            str(metrics / f"{name}.json"),
        )
    call(
        sys.executable,
        "scripts/run_tracking_demo.py",
        "--data-dir",
        str(data_dir),
        "--output",
        str(metrics / "tracking.json"),
    )
    call(sys.executable, "scripts/build_skill_matrix.py")
    call(sys.executable, "-m", "pytest", "-q")
    call(sys.executable, "-m", "ruff", "check", "src", "scripts", "tests")


if __name__ == "__main__":
    main()
