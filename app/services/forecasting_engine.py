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
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _history_cap(history: list[float]) -> float:
    finite = sorted(max(0.0, float(value)) for value in history if math.isfinite(float(value)))
    if not finite:
        return 1.0
    quantile_index = min(len(finite) - 1, int((len(finite) - 1) * 0.995))
    return max(finite[quantile_index] * 4.0, (sum(finite) / len(finite)) * 10.0, 1.0)


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
    cap = max(float(ridge.get("prediction_cap", 0.0)), _history_cap(history))
    if ridge.get("target_transform") == "log1p":
        prediction = math.expm1(min(max(prediction, -20.0), math.log1p(cap)))
    if not math.isfinite(prediction):
        raise RuntimeError("Trained Ridge model produced a non-finite value")
    return min(max(0.0, prediction), cap)


def _next_prediction(history: list[float], target: date, artifact: dict) -> float:
    model_name = artifact["selected_model"]
    cap = _history_cap(history)
    if model_name == "seasonal_naive_7":
        return min(max(0.0, history[-7]), cap)
    if model_name == "moving_average_7":
        return min(max(0.0, sum(history[-7:]) / 7), cap)
    if model_name == "median_7":
        ordered = sorted(history[-7:])
        return min(max(0.0, ordered[len(ordered) // 2]), cap)
    if model_name == "moving_average_14":
        return min(max(0.0, sum(history[-14:]) / 14), cap)
    if model_name.startswith("ridge_"):
        return _ridge_predict(history, target, len(history), artifact)
    raise RuntimeError(f"Unsupported trained model: {model_name}")


def _raw_forecast(history: list[float], last_date: date, horizon: int, artifact: dict) -> list[dict]:
    """Generate the trained model trajectory before deployment-scale calibration."""
    mutable = list(map(float, history))
    raw: list[dict] = []
    for offset in range(1, horizon + 1):
        target = last_date + timedelta(days=offset)
        predicted = _next_prediction(mutable, target, artifact)
        mutable.append(predicted)
        raw.append({"date": target, "predicted": max(0.0, float(predicted))})
    return raw


def _weekly_ratio(history: list[float]) -> tuple[float, float]:
    recent7 = sum(max(0.0, float(value)) for value in history[-7:])
    previous7 = sum(max(0.0, float(value)) for value in history[-14:-7])
    if previous7 <= 0.0:
        return 1.0, recent7
    raw_ratio = recent7 / previous7
    # Deployment data can be on a completely different monetary scale than UCI.
    # Preserve the direction but damp extreme single-week shocks.
    clipped = min(1.40, max(0.60, raw_ratio))
    damped = 1.0 + 0.60 * (clipped - 1.0)
    return min(1.24, max(0.76, damped)), recent7


def _local_interval_error(history: list[float]) -> float:
    # One-week seasonal residuals are on the same monetary scale as the uploaded
    # organization, unlike the UCI residual quantiles stored in the base artifact.
    residuals = [float(history[i]) - float(history[i - 7]) for i in range(7, len(history))]
    residuals = residuals[-28:]
    if not residuals:
        return max(_std(history[-7:]), 1.0)
    return max(_std(residuals), sum(abs(value) for value in residuals) / len(residuals), 1.0)


def _calibrate_to_local_scale(history: list[float], raw: list[dict]) -> tuple[list[dict], dict]:
    """Calibrate a trained-model trajectory to the uploaded organization's scale.

    The Ridge artifact was trained on UCI Online Retail, so its raw absolute SAR
    level must not be copied directly to another merchant. We keep its relative
    day-to-day trajectory but blend it with the merchant's last observed weekly
    shape, then anchor the total to the merchant's recent 7-day level and trend.
    This is a deployment calibration layer, not retraining.
    """
    trend_factor, recent7_total = _weekly_ratio(history)
    recent_week = [max(0.0, float(value)) for value in history[-7:]]
    recent_week_total = sum(recent_week)
    if recent_week_total <= 0.0:
        recent_week = [1.0] * 7
        recent_week_total = 7.0

    calibrated: list[dict] = []
    local_error = _local_interval_error(history)
    week_growth = min(1.12, max(0.88, 1.0 + 0.35 * (trend_factor - 1.0)))

    for week_start in range(0, len(raw), 7):
        segment = raw[week_start:week_start + 7]
        if not segment:
            break
        week_index = week_start // 7
        target_week_total = recent7_total * trend_factor * (week_growth ** week_index)
        if len(segment) < 7:
            target_week_total *= len(segment) / 7.0

        raw_values = [max(0.0, float(item["predicted"])) for item in segment]
        raw_total = sum(raw_values)
        if raw_total <= 0.0:
            raw_weights = [1.0 / len(segment)] * len(segment)
        else:
            raw_weights = [value / raw_total for value in raw_values]

        local_slice = recent_week[:len(segment)]
        local_total = sum(local_slice)
        if local_total <= 0.0:
            local_weights = [1.0 / len(segment)] * len(segment)
        else:
            local_weights = [value / local_total for value in local_slice]

        blended = [0.35 * rw + 0.65 * lw for rw, lw in zip(raw_weights, local_weights)]
        blended_total = sum(blended) or 1.0
        blended = [value / blended_total for value in blended]

        for item, weight in zip(segment, blended):
            predicted = max(0.0, target_week_total * weight)
            calibrated.append({
                "date": item["date"],
                "predicted": predicted,
                "interval_error": local_error,
            })

    return calibrated, {
        "method": "uci_ridge_shape_plus_local_weekly_level_v1",
        "recent_7d_total": recent7_total,
        "trend_factor": trend_factor,
        "local_interval_error": local_error,
    }


def forecast(history: list[float], last_date: date, horizon: int) -> list[dict]:
    if horizon not in {7, 30}:
        raise ValueError("Only 7-day and 30-day horizons are supported")
    if len(history) < 28:
        raise ValueError("At least 28 daily observations are required")

    clean_history = [max(0.0, float(value)) for value in history]
    artifact = load_artifact()
    raw = _raw_forecast(clean_history, last_date, horizon, artifact)
    calibrated, calibration = _calibrate_to_local_scale(clean_history, raw)

    baseline = sum(clean_history[-28:]) / 28
    local_error = max(float(calibration["local_interval_error"]), 1.0)
    generated: list[dict] = []

    for item in calibrated:
        predicted = float(item["predicted"])
        decline = max(0.0, (baseline - predicted) / max(abs(baseline), 1.0))
        z_score = max(-20.0, min(20.0, (baseline - predicted) / local_error))
        probability = 1.0 / (1.0 + math.exp(-z_score))
        generated.append({
            "date": item["date"],
            "predicted": predicted,
            "lower": max(0.0, predicted - 1.64 * local_error),
            "upper": max(0.0, predicted + 1.64 * local_error),
            "baseline": baseline,
            "decline_percent": decline,
            "decline_probability": probability,
            "model_name": artifact["selected_model"],
            "model_version": artifact["version"],
            "metrics": artifact["metrics"],
            "calibration": calibration,
        })
    return generated
