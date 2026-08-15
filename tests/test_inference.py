import numpy as np
import pandas as pd
import pytest

from src.inference import (
    analyze_price_anomaly,
    get_segment_statistics,
    predict_price,
)


class FakePriceModel:
    """Small deterministic model used only for unit tests."""

    def __init__(self, predicted_price: float):
        self.predicted_price = predicted_price

    def predict(self, data):
        return np.array([np.log1p(self.predicted_price)])


class FakeIsolationPreprocessor:
    """Return a simple matrix without running real preprocessing."""

    def transform(self, data):
        return np.zeros((len(data), 1))


class FakeIsolationForest:
    """Control Isolation Forest score and flag in unit tests."""

    def __init__(self, raw_score: float = 0.30, is_outlier: bool = False):
        self.raw_score = raw_score
        self.is_outlier = is_outlier

    def score_samples(self, data):
        return np.full(len(data), -self.raw_score)

    def predict(self, data):
        label = -1 if self.is_outlier else 1
        return np.full(len(data), label)


def make_config():
    return {
        "global_residual_median": 0.0,
        "global_residual_scale": 100.0,
        "global_percentiles": {
            "p01": 10.0,
            "p10": 50.0,
            "p90": 150.0,
            "p99": 200.0,
        },
        "isolation_numeric_features": [
            "deployment_predicted_price",
            "deployment_residual_z",
        ],
        "isolation_categorical_features": [],
        "isolation_score_min": 0.30,
        "isolation_score_max": 0.40,
        "weights": {
            "residual": 0.40,
            "minmax": 0.20,
            "common_range": 0.20,
            "isolation": 0.20,
        },
        "anomaly_threshold": 50.0,
    }


def make_segment_statistics():
    return pd.DataFrame(
        [
            {
                "segment": "Honda | SH | 2020-2025",
                "residual_median": 0.0,
                "residual_scale": 100.0,
                "p01": 10.0,
                "p10": 50.0,
                "p90": 150.0,
                "p99": 200.0,
            }
        ]
    )


def make_vehicle():
    return pd.DataFrame(
        {
            "segment": ["Honda | SH | 2020-2025"],
        }
    )


def run_anomaly_check(
    *,
    predicted_price: float,
    listed_price: float,
    config=None,
    isolation_forest=None,
):
    if config is None:
        config = make_config()

    if isolation_forest is None:
        isolation_forest = FakeIsolationForest()

    return analyze_price_anomaly(
        vehicle_data=make_vehicle(),
        listed_price=listed_price,
        price_model=FakePriceModel(predicted_price),
        isolation_preprocessor=FakeIsolationPreprocessor(),
        isolation_forest=isolation_forest,
        segment_statistics=make_segment_statistics(),
        config=config,
    )


def test_predict_price_returns_original_price_scale():
    result = predict_price(
        make_vehicle(),
        FakePriceModel(predicted_price=100.0),
    )

    assert result == pytest.approx(100.0)


def test_segment_statistics_uses_matching_segment():
    result = get_segment_statistics(
        segment="Honda | SH | 2020-2025",
        segment_statistics=make_segment_statistics(),
        config=make_config(),
    )

    assert result["p10"] == 50.0
    assert result["p90"] == 150.0
    assert result["used_global_fallback"] is False


def test_unknown_segment_uses_global_fallback():
    result = get_segment_statistics(
        segment="Unknown Segment",
        segment_statistics=make_segment_statistics(),
        config=make_config(),
    )

    assert result["p10"] == 50.0
    assert result["p90"] == 150.0
    assert result["used_global_fallback"] is True


def test_normal_listing_stays_normal():
    result = run_anomaly_check(
        predicted_price=100.0,
        listed_price=95.0,
    )

    assert result["is_anomaly"] is False
    assert result["needs_manual_review"] is False
    assert result["review_status"] == "Bình thường"


def test_large_price_gap_requires_manual_review():
    result = run_anomaly_check(
        predicted_price=100.0,
        listed_price=50.0,
    )

    assert result["is_anomaly"] is False
    assert result["flag_large_price_gap"] is True
    assert result["needs_manual_review"] is True
    assert result["review_status"] == "Cần kiểm tra thủ công"


def test_score_above_threshold_is_anomalous():
    result = run_anomaly_check(
        predicted_price=100.0,
        listed_price=500.0,
    )

    assert result["anomaly_score"] >= result["anomaly_threshold"]
    assert result["is_anomaly"] is True
    assert result["review_status"] == "Bất thường"


def test_non_positive_listed_price_is_rejected():
    with pytest.raises(ValueError):
        run_anomaly_check(
            predicted_price=100.0,
            listed_price=0.0,
        )