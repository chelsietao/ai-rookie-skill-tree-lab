"""Read-only API for skills, metrics, and local churn prediction."""

from __future__ import annotations

import json
import os
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from rookie_lab import __version__
from rookie_lab.tracking.registry import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = Path(os.getenv("ROOKIE_LAB_DATA_DIR", "data/generated/ci"))
METRICS_DIR = Path(os.getenv("ROOKIE_LAB_METRICS_DIR", "artifacts/metrics"))


class ChurnRequest(BaseModel):
    age: int = Field(ge=18, le=100)
    tenure_days: int = Field(ge=0, le=10000)
    region: str
    channel: str
    income: float | None = Field(default=None, ge=0)
    visits_30d: int = Field(ge=0, le=500)
    support_tickets: int = Field(ge=0, le=100)
    loyalty_score: float | None = Field(default=None, ge=0, le=1)


app = FastAPI(
    title="AI Rookie Skill Tree Lab",
    version=__version__,
    description="Local educational API over fictional synthetic data.",
)


@app.middleware("http")
async def request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    return response


@lru_cache(maxsize=1)
def _model() -> Any:
    path = Path("artifacts/models/best_classical.joblib")
    if not path.exists():
        raise FileNotFoundError("run `make classical` first")
    return joblib.load(path)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "version": __version__, "mode": "local", "secrets_required": False}


@app.get("/skills")
def skills() -> dict[str, object]:
    path = Path("data/official/ai_rookie_active_skills.csv")
    frame = pd.read_csv(path)
    return {
        "active_nodes": len(frame),
        "ai": int(frame["path"].str.startswith("/AI").sum()),
        "rows": frame.to_dict("records"),
    }


@app.get("/metrics/{lab}")
def metrics(lab: str) -> dict[str, object]:
    if not lab.replace("-", "").isalnum():
        raise HTTPException(400, "invalid lab name")
    path = METRICS_DIR / f"{lab}.json"
    if not path.exists():
        raise HTTPException(404, "metrics not found")
    return dict(json.loads(path.read_text()))


@app.get("/runs")
def runs() -> dict[str, object]:
    path = Path("artifacts/tmp/experiment_registry.sqlite")
    if not path.exists():
        return {"count": 0, "runs": []}
    values = ExperimentRegistry(path).list()
    return {
        "count": len(values),
        "runs": [
            {
                "run_id": value.run_id,
                "lab": value.lab,
                "framework": value.framework,
                "seed": value.seed,
                "config": value.config,
                "metrics": value.metrics,
            }
            for value in values
        ],
    }


@app.get("/runs/{run_id}/artifacts")
def run_artifacts(run_id: str) -> dict[str, object]:
    if not run_id.replace("-", "").isalnum():
        raise HTTPException(400, "invalid run id")
    path = Path("artifacts/tmp/experiment_registry.sqlite")
    if not path.exists():
        return {"run_id": run_id, "artifacts": []}
    return {"run_id": run_id, "artifacts": ExperimentRegistry(path).artifacts(run_id)}


@app.post("/predict/churn")
def predict_churn(payload: ChurnRequest) -> dict[str, object]:
    try:
        model = _model()
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error
    probability = float(model.predict_proba(pd.DataFrame([payload.model_dump()]))[0, 1])
    return {"churn_probability": probability, "synthetic_demo_only": True, "financial_advice": False}
