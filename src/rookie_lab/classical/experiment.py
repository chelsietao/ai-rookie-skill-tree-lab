"""Train, evaluate, cross-validate, and compare the classical model zoo."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from scipy.special import expit
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from rookie_lab.classical.models import _xgboost, classification_models, regression_models
from rookie_lab.classical.preprocessing import feature_target, preprocessor


def _classification_metrics(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(x)[:, 1]
    else:
        # A monotonic sigmoid preserves AUC ordering while placing an SVC margin
        # on a probability-like scale without the extra Platt-calibration fit.
        score = expit(model.decision_function(x))
    prediction = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }


def _regression_metrics(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    prediction = model.predict(x)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
    }


def run(data_dir: Path, profile: str, seed: int = 20260820) -> dict[str, object]:
    frame = pd.read_csv(data_dir / "customers.csv.gz")
    x_class, y_class = feature_target(frame, "churned")
    x_train, x_test, y_train, y_test = train_test_split(
        x_class, y_class, test_size=0.25, stratify=y_class, random_state=seed
    )
    results_class: dict[str, dict[str, float]] = {}
    fitted: dict[str, Any] = {}
    for name, estimator in classification_models(seed, profile).items():
        model = Pipeline([("preprocess", preprocessor()), ("model", estimator)])
        started = time.perf_counter()
        model.fit(x_train, y_train)
        metrics = _classification_metrics(model, x_test, y_test)
        metrics["fit_seconds"] = time.perf_counter() - started
        results_class[name] = metrics
        fitted[name] = model

    x_reg, y_reg = feature_target(frame, "annual_spend")
    xr_train, xr_test, yr_train, yr_test = train_test_split(
        x_reg, y_reg, test_size=0.25, random_state=seed
    )
    results_reg: dict[str, dict[str, float]] = {}
    for name, estimator in regression_models(seed, profile).items():
        model = Pipeline([("preprocess", preprocessor()), ("model", estimator)])
        started = time.perf_counter()
        model.fit(xr_train, yr_train)
        metrics = _regression_metrics(model, xr_test, yr_test)
        metrics["fit_seconds"] = time.perf_counter() - started
        results_reg[name] = metrics

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    logistic = Pipeline(
        [
            ("preprocess", preprocessor()),
            ("model", clone(classification_models(seed, profile)["logistic_regression"])),
        ]
    )
    cv_auc = cross_val_score(logistic, x_class, y_class, cv=cv, scoring="roc_auc", n_jobs=1)

    transformed = preprocessor().fit_transform(x_train)
    before_rate = float(y_train.mean())
    _, balanced_y = RandomOverSampler(random_state=seed).fit_resample(transformed, y_train)
    imbalance_demo = {
        "before_positive_rate": before_rate,
        "after_positive_rate": float(balanced_y.mean()),
    }

    best_name = max(results_class, key=lambda key: results_class[key]["roc_auc"])
    models_dir = Path("artifacts/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], models_dir / "best_classical.joblib")
    return {
        "classification": results_class,
        "regression": results_reg,
        "cross_validation": {"logistic_roc_auc_mean": float(np.mean(cv_auc)), "folds": cv_auc.tolist()},
        "class_imbalance": imbalance_demo,
        "best_classifier": best_name,
        "xgboost_backend": _xgboost(seed, 1)[2],
        "test_rows": len(x_test),
    }
