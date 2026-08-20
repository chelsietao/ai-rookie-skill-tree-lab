from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Export small reviewable samples from generated data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated/ci"))
    parser.add_argument("--output", type=Path, default=Path("data/samples"))
    parser.add_argument("--rows", type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in (
        "customers.csv.gz",
        "transactions.csv.gz",
        "product_views.csv.gz",
        "support_texts.csv.gz",
        "graph_edges.csv.gz",
    ):
        frame = pd.read_csv(args.data_dir / filename).head(args.rows)
        frame.to_csv(args.output / filename.replace(".gz", ""), index=False)
    print(f"exported five {args.rows}-row samples to {args.output}")


if __name__ == "__main__":
    main()
