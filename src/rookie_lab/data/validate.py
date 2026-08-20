"""Data contracts for the generated retail world."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate(data_dir: Path) -> dict[str, object]:
    required = {
        "customers.csv.gz",
        "transactions.csv.gz",
        "product_views.csv.gz",
        "support_texts.csv.gz",
        "graph_edges.csv.gz",
        "sequences.npz",
        "product_images.npz",
        "detection_labels.json",
        "manifest.json",
    }
    missing = sorted(name for name in required if not (data_dir / name).exists())
    if missing:
        raise FileNotFoundError(f"missing generated files: {missing}")
    customers = pd.read_csv(data_dir / "customers.csv.gz")
    transactions = pd.read_csv(data_dir / "transactions.csv.gz")
    views = pd.read_csv(data_dir / "product_views.csv.gz")
    texts = pd.read_csv(data_dir / "support_texts.csv.gz")
    sequences = np.load(data_dir / "sequences.npz")
    images = np.load(data_dir / "product_images.npz")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    checks = {
        "customer_ids_unique": bool(customers["customer_id"].is_unique),
        "transaction_ids_unique": bool(transactions["transaction_id"].is_unique),
        "view_ids_unique": bool(views["view_id"].is_unique),
        "view_conversion_binary": set(views["converted"].unique()) <= {0, 1},
        "view_customer_fk": bool(views["customer_id"].isin(customers["customer_id"]).all()),
        "transaction_customer_fk": bool(
            transactions["customer_id"].isin(customers["customer_id"]).all()
        ),
        "churn_binary": set(customers["churned"].unique()) <= {0, 1},
        "anomaly_binary": set(transactions["is_anomaly"].unique()) <= {0, 1},
        "missingness_present": bool(customers[["income", "loyalty_score"]].isna().any().all()),
        "sequence_shapes_match": len(sequences["x"]) == len(sequences["y"]),
        "image_shapes_match": len(images["x"]) == len(images["y"]),
        "text_nonempty": bool(texts["text"].str.len().gt(0).all()),
        "manifest_fictional": manifest["fictional"] is True,
        "no_real_pii": manifest["contains_real_pii"] is False,
        "checksums_complete": len(manifest["sha256"]) == 8,
    }
    if not all(checks.values()):
        raise ValueError({name: passed for name, passed in checks.items() if not passed})
    return {"status": "valid", "checks": checks, "rows": manifest["rows"]}
