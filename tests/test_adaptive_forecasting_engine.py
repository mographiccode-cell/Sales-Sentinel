from __future__ import annotations

from datetime import date

import pytest

from app.services.adaptive_forecasting_engine import MODEL_VERSION, forecast


def test_requires_28_days():
    with pytest.raises(ValueError):
        forecast([100.0] * 27, date(2026, 1, 1), 7)


def test_rejects_unsupported_horizon():
    with pytest.raises(ValueError):
        forecast([100.0] * 56, date(2026, 1, 1), 14)


def test_adaptive_forecast_uses_merchant_backtest_and_valid_intervals():
    history = []
    weekly = [100.0, 120.0, 140.0, 160.0, 180.0, 150.0, 130.0]
    for week in range(10):
        growth = 1.0 + 0.01 * week
        history.extend([value * growth for value in weekly])

    result = forecast(history, date(2026, 7, 1), 7)
    assert len(result) == 7
    assert result[0]["model_version"] == MODEL_VERSION
    assert result[0]["model_name"] in {
        "seasonal_naive_7", "moving_average_7", "moving_average_14", "weekly_trend_7"
    }
    metrics = result[0]["metrics"]
    assert metrics["selection_metric"] == "merchant_rolling_wape"
    assert metrics["evidence_scope"] == "uploaded_merchant_history"
    assert metrics["backtest_points"] > 0
    assert 0 <= metrics["wape"] < 1
    assert len(metrics["candidate_wape"]) == 4
    assert all(item["predicted"] >= 0 for item in result)
    assert all(item["lower"] <= item["predicted"] <= item["upper"] for item in result)
    # The point forecaster must never claim validated decline probability.
    assert all(item["decline_probability"] == 0.0 for item in result)


def test_model_selection_prefers_weekly_pattern_when_it_is_best():
    pattern = [80.0, 100.0, 120.0, 140.0, 160.0, 110.0, 90.0]
    history = pattern * 12
    result = forecast(history, date(2026, 7, 1), 30)
    assert result[0]["model_name"] == "seasonal_naive_7"
    assert result[0]["metrics"]["wape"] == pytest.approx(0.0)
    assert len(result) == 30


def test_recent_regime_selection_is_not_bound_to_legacy_uci_artifact():
    # A flat merchant should be forecast locally without loading or depending on
    # the legacy UCI point-forecast artifact.
    history = [2500.0] * 70
    result = forecast(history, date(2026, 7, 1), 7)
    assert result[0]["predicted"] == pytest.approx(2500.0)
    assert result[0]["metrics"]["wape"] == pytest.approx(0.0)
    assert "UCI" not in result[0]["model_version"].upper()
