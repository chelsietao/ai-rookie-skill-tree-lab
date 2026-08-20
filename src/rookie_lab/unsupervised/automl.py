"""Small transparent AutoML exercise using grid, random, and Bayesian search."""

from __future__ import annotations

from pathlib import Path

import optuna
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from rookie_lab.classical.preprocessing import feature_target, preprocessor


def run(data_dir: Path, profile: str, seed: int = 20260820) -> dict[str, object]:
    frame = pd.read_csv(data_dir / "customers.csv.gz")
    x, y = feature_target(frame, "churned")
    if profile == "ci":
        x, y = x.iloc[:600], y.iloc[:600]
    grid = GridSearchCV(
        Pipeline([("preprocess", preprocessor()), ("model", LogisticRegression(max_iter=500))]),
        {"model__C": [0.1, 1.0, 10.0]},
        scoring="roc_auc",
        cv=3,
    ).fit(x, y)
    random = RandomizedSearchCV(
        Pipeline([("preprocess", preprocessor()), ("model", ExtraTreesClassifier(random_state=seed))]),
        {"model__n_estimators": [30, 60, 100], "model__max_depth": [3, 5, 8, None]},
        n_iter=4 if profile == "ci" else 8,
        scoring="roc_auc",
        cv=3,
        random_state=seed,
    ).fit(x, y)

    def objective(trial: optuna.Trial) -> float:
        model = Pipeline(
            [
                ("preprocess", preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=trial.suggest_int("trees", 20, 80),
                        max_depth=trial.suggest_int("depth", 3, 10),
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
        return float(cross_val_score(model, x, y, cv=3, scoring="roc_auc").mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=3 if profile == "ci" else 8, show_progress_bar=False)
    return {
        "grid_search": {"score": float(grid.best_score_), "params": grid.best_params_},
        "random_search": {"score": float(random.best_score_), "params": random.best_params_},
        "optuna_tpe": {"score": float(study.best_value), "params": study.best_params},
    }
