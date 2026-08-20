import numpy as np
import pandas as pd
import pytest

from rookie_lab.classical.experiment import _expected_calibration_error
from rookie_lab.classical.preprocessing import feature_target, preprocessor


def test_preprocessor_imputes_and_encodes() -> None:
    frame = pd.DataFrame(
        {
            "age": [20, 30],
            "tenure_days": [10, 20],
            "region": ["north", "south"],
            "channel": ["organic", "search"],
            "income": [100.0, np.nan],
            "visits_30d": [4, 7],
            "support_tickets": [0, 1],
            "loyalty_score": [np.nan, 0.8],
            "churned": [0, 1],
        }
    )
    x, y = feature_target(frame, "churned")
    transformed = preprocessor().fit_transform(x)
    assert transformed.shape[0] == len(y)
    assert np.isfinite(transformed).all()


def test_expected_calibration_error_extremes() -> None:
    labels = pd.Series([0, 0, 1, 1])
    assert _expected_calibration_error(labels, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0)
    assert _expected_calibration_error(labels, np.array([0.9, 0.9, 0.1, 0.1])) > 0.7


def test_feature_target_removes_requested_label() -> None:
    frame = pd.DataFrame(
        {
            "age": [20, 30],
            "tenure_days": [10, 20],
            "region": ["north", "south"],
            "channel": ["organic", "search"],
            "income": [100.0, 200.0],
            "visits_30d": [4, 7],
            "support_tickets": [0, 1],
            "loyalty_score": [0.2, 0.8],
            "target": [0, 1],
        }
    )
    features, target = feature_target(frame, "target")
    assert "target" not in features
    assert target.tolist() == [0, 1]
