from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "models" / "sales_forecast.json"


def load_artifact() -> dict:
    if not ARTIFACT.exists():
        raise RuntimeError("The trained model artifact is missing. Run scripts/build_uci_online_retail.py")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _ridge_predict(history: list[float], target_date: date, trend: int, model: dict) -> float:
    values = {
        "lag_1": history[-1], "lag_2": history[-2], "lag_3": history[-3],
        "lag_7": history[-7], "lag_14": history[-14], "lag_28": history[-28],
        "rolling_7": sum(history[-7:]) / 7,
        "rolling_28": sum(history[-28:]) / 28,
        "weekday": target_date.weekday(), "month": target_date.month, "trend": trend,
    }
    ridge = model["ridge"]
    scaled = [(values[name] - mean) / (scale or 1.0) for name, mean, scale in zip(ridge["features"], ridge["mean"], ridge["scale"])]
    return max(0.0, ridge["intercept"] + sum(coef * value for coef, value in zip(ridge["coefficients"], scaled)))


def forecast(history: list[float], last_date: date, horizon: int) -> list[dict]:
    if horizon not in {7, 30}:
        raise ValueError("Only 7-day and 30-day horizons are supported")
    if len(history) < 28:
        raise ValueError("At least 28 daily observations are required")
    artifact = load_artifact()
    model_name = artifact["selected_model"]
    lower_error = artifact["residual_quantiles"]["lower"]
    upper_error = artifact["residual_quantiles"]["upper"]
    error_std = max(artifact["residual_quantiles"]["std"], 1.0)
    baseline = sum(history[-28:]) / 28
    generated: list[dict] = []
    mutable = list(history)
    for offset in range(1, horizon + 1):
        target = last_date + timedelta(days=offset)
        if model_name == "seasonal_naive_7":
            predicted = mutable[-7]
        elif model_name == "moving_average_7":
            predicted = sum(mutable[-7:]) / 7
        else:
            predicted = _ridge_predict(mutable, target, len(mutable), artifact)
        mutable.append(predicted)
        decline = max(0.0, (baseline - predicted) / max(abs(baseline), 1.0))
        z = (baseline - predicted) / error_std
        probability = 1.0 / (1.0 + math.exp(-z))
        generated.append({
            "date": target, "predicted": predicted,
            "lower": max(0.0, predicted + lower_error), "upper": max(0.0, predicted + upper_error),
            "baseline": baseline, "decline_percent": decline,
            "decline_probability": probability, "model_name": model_name,
            "model_version": artifact["version"], "metrics": artifact["metrics"],
        })
    return generated
