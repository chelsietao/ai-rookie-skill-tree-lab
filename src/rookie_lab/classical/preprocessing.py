"""Leakage-safe preprocessing for missing, numeric, categorical, and imbalanced data."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = [
    "age",
    "tenure_days",
    "region",
    "channel",
    "income",
    "visits_30d",
    "support_tickets",
    "loyalty_score",
]
NUMERIC = ["age", "tenure_days", "income", "visits_30d", "support_tickets", "loyalty_score"]
CATEGORICAL = ["region", "channel"]


def feature_target(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return frame[FEATURES].copy(), frame[target].copy()


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer([("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)])
