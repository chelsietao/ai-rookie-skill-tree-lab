import numpy as np
import pandas as pd

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
