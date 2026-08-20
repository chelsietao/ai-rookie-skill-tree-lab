"""Build six executable notebooks for the software-engineering half of the skill tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_learning_notebooks import SETUP, code, md


def notebook(title: str, purpose: str, cells: list[dict[str, object]]) -> dict[str, object]:
    return {
        "cells": [
            md(f"# {title}\n\n{purpose}"),
            SETUP,
            *cells,
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"name": title, "provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


BOOKS = {
    "00_start_here.ipynb": notebook(
        "00 · Start Here — evidence, environments, and reproducibility",
        "Use the repository as a research portfolio: inspect evidence, run the quick profile, and learn "
        "what the project does and does not claim.",
        [
            md("""## Learning contract

Each chapter must leave six kinds of evidence: implementation, configuration, automated test,
metric, figure/artifact, and limitation. A keyword in a README is not treated as a learned skill."""),
            code("""
            from pathlib import Path
            required = ["src", "tests", "configs", "notebooks", "artifacts/metrics", "reports"]
            print({name: (ROOT / name).exists() for name in required})
            """),
            md("""## Local versus Colab

Local mode uses the checked-out code and synthetic files. Colab clones the public repository and
installs the declared extras. Neither path asks for a secret. Generated data never contains real
customers, investment advice, or production credentials."""),
            code("""
            import platform
            import rookie_lab
            print({"package": rookie_lab.__version__, "python": platform.python_version(),
                   "platform": platform.platform(), "torch": torch.__version__})
            """),
            md("""## Two profiles

`ci` is a fast behavioral check. `portfolio` uses the requested 20k customers, 100k transactions,
50k product views, 10k sequences, 5k texts, and 5k images. Quick success never substitutes for the
committed full-profile evidence."""),
            code("""
            import yaml
            profiles = {}
            for name in ("ci", "portfolio"):
                profiles[name] = yaml.safe_load(Path(f"configs/data/{name}.yaml").read_text())
            display(pd.DataFrame(profiles).T)
            """),
            md("""## Determinism contract

Seeds, configuration, data checksums, dependency versions, elapsed time, and memory are recorded.
Exact floating-point identity can still vary by hardware and library build; tolerance-based tests
and metric ranges are more honest than promising bit-for-bit equality everywhere."""),
            code("""
            quick = yaml.safe_load(Path("configs/experiments/quick.yaml").read_text())
            full = yaml.safe_load(Path("configs/experiments/full.yaml").read_text())
            display(pd.DataFrame({"quick": quick, "full": full}).T)
            """),
            md("""## Inspect committed evidence"""),
            code("""
            metric_rows = []
            for path in sorted(Path("artifacts/metrics").glob("*.json")):
                payload = json.loads(path.read_text())
                metric_rows.append({"lab": path.stem, "keys": ", ".join(payload)})
            display(pd.DataFrame(metric_rows))
            """),
            md("""## Recommended route

Go in numerical order. In each notebook first leave `RUN_TRAINING=False` to understand the saved
evidence, then enable it for the CI profile. Run `rookie-lab reproduce --profile portfolio` only
after the chapter mechanics are clear."""),
            code("""
            books = sorted(path.name for path in Path("notebooks").glob("*.ipynb"))
            display(pd.Series(books, name="learning_path").to_frame())
            """),
            md("""## Exit check

Explain the difference between a reproducible experiment and a deterministic algorithm; explain
why the portfolio profile is evidence while the CI profile is a regression test; identify where
data lineage, environment, limitations, and security boundaries are recorded."""),
        ],
    ),
    "08_tracking_and_sql.ipynb": notebook(
        "08 · Experiment tracking, artifacts, SQLite, MySQL, and MSSQL",
        "Build a local registry, query its normalized records, attach artifacts, and compare SQL dialects.",
        [
            md("""## Why tracking is part of modeling

A metric without its seed, data version, configuration, environment, and artifact cannot be audited.
This project keeps the registry local and transparent so its schema can be inspected directly."""),
            code("""
            import time
            from rookie_lab.tracking.metadata import enrich_metrics
            if not (DATA_DIR / "manifest.json").exists():
                subprocess.run([sys.executable, "scripts/prepare_data.py", "--profile", "ci"], check=True)
            sample = enrich_metrics({"roc_auc": 0.71}, lab="notebook", profile="ci",
                                    data_dir=DATA_DIR, started=time.perf_counter(),
                                    config_path="configs/experiments/quick.yaml")
            display(pd.Series(sample))
            """),
            md("""## Create an isolated SQLite registry"""),
            code("""
            import tempfile
            from rookie_lab.tracking.registry import ExperimentRegistry, ExperimentRun
            registry_path = Path(tempfile.mkdtemp()) / "registry.sqlite"
            registry = ExperimentRegistry(registry_path)
            registry.log(ExperimentRun("nb-001", "classical", "sklearn", 20260820,
                                       {"model": "logistic"}, {"roc_auc": 0.71}))
            registry.log_artifact("nb-001", "figure", "artifacts/figures/classical.png", "demo-sha")
            print(registry.list(), registry.artifacts("nb-001"))
            """),
            md("""## Query the physical schema

The API wraps the registry, but SQL remains useful for auditing row shape and query plans."""),
            code("""
            import sqlite3
            with sqlite3.connect(registry_path) as connection:
                display(pd.read_sql_query("SELECT run_id, lab, framework, seed FROM experiment_runs", connection))
                display(pd.read_sql_query("SELECT * FROM artifacts", connection))
            """),
            md("""## Primary keys, foreign keys, and indexes

The retail schema separates entities and events. Foreign keys protect customer/product references;
indexes accelerate common lookup paths. They do not replace application validation or monitoring."""),
            code("""
            schemas = {name: Path(f"sql/{name}/schema.sql").read_text()
                       for name in ("sqlite", "mysql", "mssql")}
            for name, sql in schemas.items():
                print(name, {"tables": sql.upper().count("CREATE TABLE"),
                             "indexes": sql.upper().count("CREATE INDEX")})
            """),
            md("""## Dialect differences"""),
            code("""
            from rookie_lab.database.dialects import compiled_schemas
            compiled = compiled_schemas()
            display(pd.DataFrame({name: {"DDL preview": ddl[:120]} for name, ddl in compiled.items()}).T)
            """),
            md("""## Security boundary

Passwords are read from environment variables at runtime and are never embedded in a URL example,
notebook, committed config, or test fixture. SQLite is the fully exercised local path; MySQL has a
service CI smoke test; MSSQL is schema-validated but requires an external service for integration."""),
            code("""
            print(Path(".env.example").read_text())
            """),
            md("""## Exercises

Add a unique constraint that prevents duplicate view events; compare an `EXPLAIN` plan before and
after an index; decide whether an artifact should be deleted when its run is removed; design an
append-only alternative to `INSERT OR REPLACE`."""),
        ],
    ),
    "09_api_cli_and_serialization.ipynb": notebook(
        "09 · FastAPI, CLI contracts, validation, and serialization",
        "Exercise the application boundary without starting a public server or requiring credentials.",
        [
            md("""## Boundary design

The package contains reusable domain code. The CLI is a human/automation boundary; FastAPI is an
HTTP boundary. Both delegate instead of duplicating model and data logic."""),
            code("""
            import subprocess
            cli = Path(sys.executable).parent / "rookie-lab"
            help_result = subprocess.run([str(cli), "--help"],
                                         capture_output=True, text=True, check=True)
            print(help_result.stdout)
            """),
            md("""## Test the API in-process

`TestClient` runs routing, validation, middleware, and serialization without binding a network port."""),
            code("""
            from fastapi.testclient import TestClient
            from rookie_lab.serving.api import app
            client = TestClient(app)
            response = client.get("/health", headers={"X-Request-ID": "notebook-demo"})
            print(response.status_code, response.json(), response.headers["X-Request-ID"])
            """),
            md("""## Input validation and error semantics"""),
            code("""
            invalid = client.post("/predict/churn", json={"age": 8})
            missing = client.get("/metrics/does-not-exist")
            unsafe = client.get("/metrics/../secret")
            print({"invalid": invalid.status_code, "missing": missing.status_code,
                   "unsafe_path": unsafe.status_code})
            """),
            md("""## Read-only evidence endpoints"""),
            code("""
            skills = client.get("/skills").json()
            print({"official_nodes": skills["active_nodes"], "ai_nodes": skills["ai"]})
            print(client.get("/runs").json()["count"])
            """),
            md("""## JSON and binary serialization

JSON is inspectable and language-neutral. Joblib is convenient for Python pipelines but must never
load an untrusted file because pickle-compatible formats can execute code."""),
            code("""
            import tempfile
            from rookie_lab.software.serialization import dump_payload, load_payload
            payload = {"model": "demo", "metrics": [0.6, 0.7]}
            serialization_dir = Path(tempfile.mkdtemp())
            for suffix in ("json", "yaml"):
                path = serialization_dir / f"payload.{suffix}"
                dump_payload(payload, path)
                print(suffix, load_payload(path))
            """),
            md("""## Production gaps

The API binds to localhost by default. It lacks authentication, rate limiting, a feature store,
drift monitoring, rollout policy, and a production model registry. It is a local software contract,
not a claim of production readiness or financial suitability."""),
            code("""
            schema = client.get("/openapi.json").json()
            rows = [{"path": path, "methods": ", ".join(item)}
                    for path, item in schema["paths"].items()]
            display(pd.DataFrame(rows).set_index("path"))
            """),
            md("""## Exercises

Add an API version prefix; return a structured model-unavailable error; write a latency budget test;
design authentication without committing a key; explain why Pydantic validation cannot prevent
distribution shift."""),
        ],
    ),
    "10_testing_ci_and_security.ipynb": notebook(
        "10 · Testing pyramid, CI, logging, and secret safety",
        "Inspect and run the quality gates that turn demos into maintainable evidence.",
        [
            md("""## Testing pyramid

Unit tests isolate numerical and data contracts. Integration tests cross the FastAPI boundary.
Safety tests scan tracked source patterns. Workflow smoke tests add a real MySQL service and a
scheduled full-profile run."""),
            code("""
            tests = []
            for path in sorted(Path("tests").rglob("test_*.py")):
                tests.append({"layer": path.parent.name, "file": path.name,
                              "tests": path.read_text().count("def test_")})
            display(pd.DataFrame(tests))
            """),
            md("""## Run the local suite"""),
            code("""
            result = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                                    capture_output=True, text=True)
            print(result.stdout[-2000:])
            assert result.returncode == 0, result.stderr
            """),
            md("""## Static quality gate"""),
            code("""
            result = subprocess.run([sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"],
                                    capture_output=True, text=True)
            print(result.stdout or "Ruff passed")
            assert result.returncode == 0, result.stderr
            """),
            md("""## CI is executable documentation"""),
            code("""
            print(Path(".github/workflows/lint-test.yml").read_text())
            print(Path(".github/workflows/mysql-smoke.yml").read_text())
            """),
            md("""## Structured logging

Logs need time, level, event, and correlation identifiers. They must not include raw text bodies,
credentials, or customer identifiers. This fictional dataset still practices the same boundary."""),
            code("""
            from rookie_lab.software.logging import get_logger
            logger = get_logger("notebook")
            logger.info("notebook_quality_gate run_id=safe-demo")
            """),
            md("""## Secret scan and false confidence

Pattern tests catch common mistakes but not every secret format or leaked value in Git history.
Repository rules, short-lived credentials, least privilege, rotation, and provider-side scanning are
still required for a real system."""),
            code("""
            secret_test = Path("tests/safety/test_no_secrets.py").read_text()
            print(secret_test)
            assert ".env" in Path(".gitignore").read_text()
            """),
            md("""## Exercises

Write a failing test before a bug fix; add a property test for bounding boxes; distinguish a flaky
test from nondeterministic training; specify which checks block pull requests and which run nightly."""),
        ],
    ),
    "11_full_reproducibility.ipynb": notebook(
        "11 · Full reproducibility audit",
        "Audit the complete experiment chain and optionally reproduce it without silently hiding failures.",
        [
            md("""## Reproduction is a chain

Configuration generates versioned data; validation checks the data contract; experiments record
metadata and artifacts; tests verify behavior; reports summarize evidence and preserve limitations."""),
            code("""
            manifest_path = Path("data/generated/portfolio/manifest.json")
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                print(manifest["dataset_version"], manifest["rows"])
            else:
                print("Portfolio data is generated locally by the reproduction command.")
            """),
            md("""## Verify committed metric metadata"""),
            code("""
            audit = []
            for path in sorted(Path("artifacts/metrics").glob("*.json")):
                payload = json.loads(path.read_text())
                run = payload.get("_run", {})
                audit.append({"lab": path.stem, "profile": run.get("profile"),
                              "seed": run.get("seed"), "dataset": run.get("dataset_version"),
                              "elapsed": run.get("elapsed_seconds")})
            display(pd.DataFrame(audit))
            """),
            md("""## Artifact integrity"""),
            code("""
            import hashlib
            artifact_rows = []
            for path in sorted(Path("artifacts/figures").glob("*portfolio*")):
                artifact_rows.append({"file": path.name, "bytes": path.stat().st_size,
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]})
            display(pd.DataFrame(artifact_rows))
            """),
            md("""## Optional full run

The flag defaults to false so merely opening the notebook cannot start a long job. Set it only after
checking disk, CPU budget, and the profile. Failures are raised, not converted into green results."""),
            code(
                """
            RUN_FULL_REPRODUCTION = False
            if RUN_FULL_REPRODUCTION:
                subprocess.run([sys.executable, "scripts/reproduce_all.py", "--profile", "portfolio"],
                               check=True)
            else:
                print("Skipped by explicit safety switch; committed portfolio evidence remains inspectable.")
            """,
                ["parameters"],
            ),
            md("""## Reproducibility report"""),
            code("""
            subprocess.run([sys.executable, "scripts/build_reproducibility_report.py"], check=True)
            print(Path("reports/reproducibility-report.md").read_text())
            """),
            md("""## Expected variation

Wall time, resident memory, TensorFlow kernels, parallel tree training, and accelerator behavior can
vary. Dataset checksums and configuration should remain stable; metrics should remain within justified
tolerances. Investigate a deviation before accepting or suppressing it."""),
            code("""
            print(Path("reports/limitations.md").read_text())
            """),
            md("""## Exercises

Reproduce on a second machine; compare environment captures; create an acceptance range for ROC-AUC;
change one seed and quantify variance; explain why committing only a model file is not reproducibility."""),
        ],
    ),
    "12_skill_matrix_resume_interview.ipynb": notebook(
        "12 · Skill evidence, résumé bullets, and interview defense",
        "Map every official node to verifiable evidence and practice making claims no stronger than the results.",
        [
            md("""## Coverage rule

The official page contains 114 active nodes: 84 under AI and 30 under software engineering. Closely
related labels are normalized for implementation, but the official wording is retained for audit."""),
            code("""
            skills = pd.read_csv("data/official/ai_rookie_active_skills.csv")
            display(skills.groupby(skills.path.str.split("/").str[1]).size().rename("active_nodes"))
            """),
            md("""## Rebuild the evidence matrix"""),
            code("""
            subprocess.run([sys.executable, "scripts/build_skill_matrix.py"], check=True)
            matrix_text = Path("docs/official-skill-matrix.md").read_text()
            print("matrix rows", matrix_text.count("\\n|"), "characters", len(matrix_text))
            print("\\n".join(matrix_text.splitlines()[:12]))
            """),
            md("""## Sample an interview trail"""),
            code("""
            terms = ["CNN", "MySQL", "BERT", "GCN", "PyTorch", "precision"]
            rows = [line for line in matrix_text.splitlines()
                    if any(term.lower() in line.lower() for term in terms)]
            display(pd.Series(rows[:20], name="evidence trail").to_frame())
            """),
            md("""## Result-aware résumé writing

Use: action + technical scope + evaluation + result + limitation. Never say “production-grade” from
a localhost API, “real customer impact” from synthetic data, or “state of the art” from a toy model."""),
            code("""
            classical = json.loads(Path("artifacts/metrics/classical.json").read_text())
            best_name, best = max(classical["classification"].items(), key=lambda item: item[1]["roc_auc"])
            bullet = (f"Built a leakage-safe {best_name} churn benchmark across a multi-model suite; "
                      f"measured ROC-AUC {best['roc_auc']:.3f}, PR-AUC {best['pr_auc']:.3f}, "
                      "calibration, errors, and CPU latency on versioned synthetic retail data.")
            print(bullet)
            """),
            md("""## Defend negative results"""),
            code("""
            graph = json.loads(Path("artifacts/metrics/graph.json").read_text())
            display(pd.Series({"GCN link ROC-AUC": graph["link_prediction"]["gcn_roc_auc"],
                               "degree baseline": graph["link_prediction"]["degree_popularity_baseline_roc_auc"]}))
            """),
            md("""A negative comparison is useful when the protocol is fair and preserved. Explain the likely
causes, propose the next discriminating experiment, and avoid retroactively changing the metric."""),
            code("""
            print(Path("docs/resume-guide.md").read_text())
            print(Path("docs/interview-guide.md").read_text())
            """),
            md("""## Mock interview prompts

1. Where can leakage enter the classical pipeline?
2. Why can high accuracy coexist with poor recall?
3. Why is the graph evaluation transductive?
4. What does the transfer-learning proxy fail to prove?
5. Which API controls are still missing?
6. Which single experiment would most strengthen this portfolio, and why?"""),
        ],
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("notebooks"))
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.check:
        stale = []
        for name, payload in BOOKS.items():
            expected = json.dumps(payload, indent=1, ensure_ascii=False) + "\n"
            if not (args.check / name).exists() or (args.check / name).read_text() != expected:
                stale.append(name)
        if stale:
            raise SystemExit(f"system notebooks are stale or missing: {', '.join(stale)}")
        print(f"verified {len(BOOKS)} system notebooks in {args.check}")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in BOOKS.items():
        (args.output / name).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(BOOKS)} system notebooks to {args.output}")


if __name__ == "__main__":
    main()
