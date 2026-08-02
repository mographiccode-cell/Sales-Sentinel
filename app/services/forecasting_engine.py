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
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    if not artifact.get("selected_model"):
        raise RuntimeError("The trained model artifact is empty or invalid")
    return artifact


def _feature_values(history: list[float], target_date: date, trend: int) -> dict[str, float]:
    return {
        "lag_1": history[-1], "lag_2": history[-2], "lag_3": history[-3],
        "lag_7": history[-7], "lag_14": history[-14], "lag_21": history[-21], "lag_28": history[-28],
        "rolling_7": sum(history[-7:]) / 7,
        "rolling_14": sum(history[-14:]) / 14,
        "rolling_28": sum(history[-28:]) / 28,
        "std_7": _std(history[-7:]), "std_14": _std(history[-14:]),
        "weekday_sin": math.sin(2 * math.pi * target_date.weekday() / 7),
        "weekday_cos": math.cos(2 * math.pi * target_date.weekday() / 7),
        "month_sin": math.sin(2 * math.pi * (target_date.month - 1) / 12),
        "month_cos": math.cos(2 * math.pi * (target_date.month - 1) / 12),
        "trend": float(trend),
    }


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _ridge_predict(history: list[float], target_date: date, trend: int, artifact: dict) -> float:
    ridge = artifact["ridge"]
    values = _feature_values(history, target_date, trend)
    scaled = [
        (values[name] - mean) / (scale or 1.0)
        for name, mean, scale in zip(ridge["features"], ridge["mean"], ridge["scale"])
    ]
    prediction = ridge["intercept"] + sum(
        coefficient * value for coefficient, value in zip(ridge["coefficients"], scaled)
    )
    if ridge.get("target_transform") == "log1p":
        prediction = math.expm1(prediction)
    return max(0.0, prediction)


def _next_prediction(history: list[float], target: date, artifact: dict) -> float:
    model_name = artifact["selected_model"]
    if model_name == "seasonal_naive_7":
        return max(0.0, history[-7])
    if model_name == "moving_average_7":
        return max(0.0, sum(history[-7:]) / 7)
    if model_name == "median_7":
        ordered = sorted(history[-7:])
        return max(0.0, ordered[len(ordered) // 2])
    if model_name == "moving_average_14":
        return max(0.0, sum(history[-14:]) / 14)
    if model_name.startswith("ridge_"):
        return _ridge_predict(history, target, len(history), artifact)
    raise RuntimeError(f"Unsupported trained model: {model_name}")


def forecast(history: list[float], last_date: date, horizon: int) -> list[dict]:
    if horizon not in {7, 30}:
        raise ValueError("Only 7-day and 30-day horizons are supported")
    if len(history) < 28:
        raise ValueError("At least 28 daily observations are required")
    artifact = load_artifact()
    lower_error = float(artifact["residual_quantiles"]["lower"])
    upper_error = float(artifact["residual_quantiles"]["upper"])
    error_std = max(float(artifact["residual_quantiles"]["std"]), 1.0)
    baseline = sum(history[-28:]) / 28
    generated: list[dict] = []
    mutable = list(map(float, history))
    for offset in range(1, horizon + 1):
        target = last_date + timedelta(days=offset)
        predicted = _next_prediction(mutable, target, artifact)
        mutable.append(predicted)
        decline = max(0.0, (baseline - predicted) / max(abs(baseline), 1.0))
        z_score = max(-20.0, min(20.0, (baseline - predicted) / error_std))
        probability = 1.0 / (1.0 + math.exp(-z_score))
        generated.append({
            "date": target,
            "predicted": predicted,
            "lower": max(0.0, predicted + lower_error),
            "upper": max(0.0, predicted + upper_error),
            "baseline": baseline,
            "decline_percent": decline,
            "decline_probability": probability,
            "model_name": artifact["selected_model"],
            "model_version": artifact["version"],
            "metrics": artifact["metrics"],
        })
    return generated
