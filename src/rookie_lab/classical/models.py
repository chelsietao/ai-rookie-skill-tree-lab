"""Required AI-rookie classification and regression estimators."""

from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _xgboost(seed: int, trees: int) -> tuple[Any, Any, str]:
    try:
        from xgboost import XGBClassifier, XGBRegressor

        classifier = XGBClassifier(
            n_estimators=trees,
            max_depth=4,
            learning_rate=0.08,
            eval_metric="logloss",
            n_jobs=2,
            random_state=seed,
        )
        regressor = XGBRegressor(
            n_estimators=trees,
            max_depth=4,
            learning_rate=0.08,
            objective="reg:squarederror",
            n_jobs=2,
            random_state=seed,
        )
        return classifier, regressor, "native-xgboost"
    except Exception:
        return (
            HistGradientBoostingClassifier(max_iter=trees, max_depth=4, random_state=seed),
            HistGradientBoostingRegressor(max_iter=trees, max_depth=4, random_state=seed),
            "sklearn-fallback-no-libomp",
        )


def classification_models(seed: int, profile: str) -> dict[str, Any]:
    trees = 35 if profile == "ci" else 100
    iterations = 35 if profile == "ci" else 100
    xgb_classifier, _, _ = _xgboost(seed, trees)
    return {
        "knn": KNeighborsClassifier(n_neighbors=7),
        "logistic_regression": LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=seed
        ),
        "svc": SVC(C=1.0, class_weight="balanced", random_state=seed),
        "decision_tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=trees, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=trees, max_depth=8, class_weight="balanced", n_jobs=-1, random_state=seed
        ),
        "xgboost": xgb_classifier,
        "catboost": CatBoostClassifier(
            iterations=iterations, depth=5, learning_rate=0.08, verbose=False, random_seed=seed
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=180, early_stopping=True, random_state=seed
        ),
    }


def regression_models(seed: int, profile: str) -> dict[str, Any]:
    trees = 35 if profile == "ci" else 100
    iterations = 35 if profile == "ci" else 100
    _, xgb_regressor, _ = _xgboost(seed, trees)
    return {
        "knn": KNeighborsRegressor(n_neighbors=7),
        "linear_regression": LinearRegression(),
        "lasso": Lasso(alpha=0.1, max_iter=2000),
        "ridge": Ridge(alpha=1.0),
        "svr": SVR(C=5.0, epsilon=0.1),
        "decision_tree": DecisionTreeRegressor(max_depth=7, random_state=seed),
        "random_forest": RandomForestRegressor(
            n_estimators=trees, max_depth=9, n_jobs=-1, random_state=seed
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=trees, max_depth=9, n_jobs=-1, random_state=seed
        ),
        "xgboost": xgb_regressor,
        "catboost": CatBoostRegressor(
            iterations=iterations, depth=5, learning_rate=0.08, verbose=False, random_seed=seed
        ),
    }
