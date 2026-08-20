from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nested(payload: dict[str, Any], *keys: str, default: Any = "n/a") -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, default=Path("artifacts/metrics"))
    parser.add_argument("--output", type=Path, default=Path("reports/reproducibility-report.md"))
    args = parser.parse_args()
    payloads = {
        path.stem: json.loads(path.read_text()) for path in sorted(args.metrics_dir.glob("*.json"))
    }
    rows = []
    for lab, payload in payloads.items():
        metadata = payload.get("_run", {})
        rows.append(
            f"| {lab} | {metadata.get('dataset_version', 'legacy run')} | "
            f"{metadata.get('seed', 'n/a')} | "
            f"{float(metadata.get('elapsed_seconds', 0)):.3f} | "
            f"{float(metadata.get('resident_memory_mb', 0)):.1f} |"
        )
    classical = payloads.get("classical", {})
    vision = payloads.get("vision", {})
    graph = payloads.get("graph", {})
    lines = [
        "# Reproducibility report",
        "",
        "This file is generated from machine-readable metrics. Regenerate it after every full run.",
        "",
        "| Lab | Dataset version | Seed | Seconds | RSS MB |",
        "|---|---|---:|---:|---:|",
        *rows,
        "",
        "## Selected checks",
        "",
        f"- Best classical classifier: {classical.get('best_classifier', 'n/a')}",
        f"- Classical diagnostic: {classical.get('diagnostic_figure', 'n/a')}",
        f"- Detector mAP@0.50: {nested(vision, 'object_detection', 'map_at_50')}",
        f"- Detector mAP@0.50:0.95: {nested(vision, 'object_detection', 'map_50_95')}",
        f"- GCN node ROC-AUC: {nested(graph, 'node_classification', 'roc_auc')}",
        f"- Logistic + degree node baseline: "
        f"{nested(graph, 'traditional_baseline', 'node_roc_auc')}",
        f"- GCN link ROC-AUC: {nested(graph, 'link_prediction', 'gcn_roc_auc')}",
        f"- Degree-popularity link baseline: "
        f"{nested(graph, 'link_prediction', 'degree_popularity_baseline_roc_auc')}",
        "",
        "Generated data and model binaries are intentionally ignored; configs, code, checksums,",
        "metrics, and figures are versioned so the artifacts can be reconstructed.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
