"""Train, evaluate, cross-validate, and compare the classical model zoo."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from scipy.special import expit
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from rookie_lab.classical.models import _xgboost, classification_models, regression_models
from rookie_lab.classical.preprocessing import feature_target, preprocessor


def _scores(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1])
    # A monotonic sigmoid preserves AUC ordering while placing an SVC margin
    # on a probability-like scale without the extra Platt-calibration fit.
    return np.asarray(expit(model.decision_function(x)))


def _expected_calibration_error(y: pd.Series, score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.clip(np.digitize(score, edges) - 1, 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            error += float(mask.mean()) * abs(
                float(y.to_numpy()[mask].mean()) - float(score[mask].mean())
            )
    return error


def _classification_metrics(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    started = time.perf_counter()
    score = _scores(model, x)
    prediction = model.predict(x)
    inference_ms = (time.perf_counter() - started) * 1000 / len(x) * 1000
    matrix = confusion_matrix(y, prediction)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
        "ece_10_bin": _expected_calibration_error(y, score),
        "confusion_matrix": matrix.tolist(),
        "inference_ms_per_1000": inference_ms,
    }


def _regression_metrics(model: Any, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    started = time.perf_counter()
    prediction = model.predict(x)
    inference_ms = (time.perf_counter() - started) * 1000 / len(x) * 1000
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
        "inference_ms_per_1000": inference_ms,
    }


def _save_diagnostics(model: Any, x: pd.DataFrame, y: pd.Series, profile: str, name: str) -> str:
    score = _scores(model, x)
    prediction = model.predict(x)
    fpr, tpr, _ = roc_curve(y, score)
    precision, recall, _ = precision_recall_curve(y, score)
    observed, predicted = calibration_curve(y, score, n_bins=8, strategy="quantile")
    matrix = confusion_matrix(y, prediction)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes[0, 0].plot(fpr, tpr, label=name)
    axes[0, 0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0, 0].set(title="ROC curve", xlabel="False-positive rate", ylabel="True-positive rate")
    axes[0, 1].plot(recall, precision)
    axes[0, 1].set(title="Precision-recall curve", xlabel="Recall", ylabel="Precision")
    image = axes[1, 0].imshow(matrix, cmap="Blues")
    for (row, column), value in np.ndenumerate(matrix):
        axes[1, 0].text(column, row, str(value), ha="center", va="center")
    axes[1, 0].set(title="Confusion matrix", xlabel="Predicted", ylabel="Actual")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)
    axes[1, 1].plot(predicted, observed, marker="o")
    axes[1, 1].plot([0, 1], [0, 1], "--", color="grey")
    axes[1, 1].set(title="Calibration", xlabel="Mean prediction", ylabel="Observed rate")
    fig.tight_layout()
    output = Path("artifacts/figures") / f"classical-diagnostics-{profile}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return str(output)


def run(data_dir: Path, profile: str, seed: int = 20260820) -> dict[str, object]:
    frame = pd.read_csv(data_dir / "customers.csv.gz")
    x_class, y_class = feature_target(frame, "churned")
    x_train, x_test, y_train, y_test = train_test_split(
        x_class, y_class, test_size=0.25, stratify=y_class, random_state=seed
    )
    results_class: dict[str, dict[str, object]] = {}
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
    diagnostic_path = _save_diagnostics(fitted[best_name], x_test, y_test, profile, best_name)
    leaderboard = sorted(
        (
            {"model": name, "roc_auc": values["roc_auc"], "pr_auc": values["pr_auc"]}
            for name, values in results_class.items()
        ),
        key=lambda row: row["roc_auc"],
        reverse=True,
    )
    return {
        "classification": results_class,
        "regression": results_reg,
        "cross_validation": {"logistic_roc_auc_mean": float(np.mean(cv_auc)), "folds": cv_auc.tolist()},
        "class_imbalance": imbalance_demo,
        "best_classifier": best_name,
        "classification_leaderboard": leaderboard,
        "diagnostic_figure": diagnostic_path,
        "xgboost_backend": _xgboost(seed, 1)[2],
        "test_rows": len(x_test),
    }
