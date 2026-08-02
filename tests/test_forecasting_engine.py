from __future__ import annotations

from datetime import date

import pytest

from app.services import forecasting_engine

FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_7", "rolling_14", "rolling_28", "std_7", "std_14",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos", "trend",
]


def artifact(model: str = "seasonal_naive_7") -> dict:
    return {
        "version": "test-v2",
        "selected_model": model,
        "metrics": {"mae": 2.0, "rmse": 3.0, "wape": 0.1, "smape": 0.1},
        "residual_quantiles": {"lower": -5.0, "upper": 8.0, "std": 10.0},
        "ridge": {
            "features": FEATURES,
            "mean": [0.0] * len(FEATURES),
            "scale": [1.0] * len(FEATURES),
            "coefficients": [0.001] * len(FEATURES),
            "intercept": 9.0,
            "alpha": 1000.0,
            "target_transform": "log1p",
        },
    }


def test_rejects_unsupported_horizon(monkeypatch):
    monkeypatch.setattr(forecasting_engine, "load_artifact", lambda: artifact())
    with pytest.raises(ValueError):
        forecasting_engine.forecast([100.0] * 40, date(2024, 1, 1), 90)


def test_requires_real_history(monkeypatch):
    monkeypatch.setattr(forecasting_engine, "load_artifact", lambda: artifact())
    with pytest.raises(ValueError):
        forecasting_engine.forecast([100.0] * 10, date(2024, 1, 1), 7)


def test_seasonal_forecast_has_empirical_intervals(monkeypatch):
    monkeypatch.setattr(forecasting_engine, "load_artifact", lambda: artifact())
    result = forecasting_engine.forecast([float(i) for i in range(1, 41)], date(2024, 1, 1), 7)
    assert len(result) == 7
    assert result[0]["predicted"] == 34.0
    assert result[0]["lower"] == 29.0
    assert result[0]["upper"] == 42.0
    assert 0 <= result[0]["decline_probability"] <= 1


def test_log_ridge_inference_is_non_negative(monkeypatch):
    monkeypatch.setattr(forecasting_engine, "load_artifact", lambda: artifact("ridge_log_1000"))
    result = forecasting_engine.forecast([100.0] * 40, date(2024, 1, 1), 30)
    assert all(item["predicted"] >= 0 for item in result)
    assert all(item["lower"] <= item["upper"] for item in result)
    assert all(item["model_name"] == "ridge_log_1000" for item in result)


def test_all_supported_baselines(monkeypatch):
    history = [float(index * 100) for index in range(1, 41)]
    for name in ("moving_average_7", "median_7", "moving_average_14"):
        monkeypatch.setattr(forecasting_engine, "load_artifact", lambda name=name: artifact(name))
        result = forecasting_engine.forecast(history, date(2024, 1, 1), 7)
        assert len(result) == 7
        assert all(item["predicted"] >= 0 for item in result)
