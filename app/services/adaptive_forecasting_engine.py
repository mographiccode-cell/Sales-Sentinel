from __future__ import annotations

import math
from datetime import date, timedelta

MODEL_VERSION = "SALES-SENTINEL-ADAPTIVE-MERCHANT-FORECAST-V2"
_CANDIDATES = (
    "seasonal_naive_7",
    "moving_average_7",
    "moving_average_14",
    "median_7",
    "median_14",
    "weekday_mean_8w",
    "weekday_median_8w",
    "seasonal_level_blend",
    "weekly_trend_7",
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def _cap(history: list[float]) -> float:
    finite = [max(0.0, float(v)) for v in history if math.isfinite(float(v))]
    if not finite:
        return 1.0
    return max(max(finite) * 3.0, _mean(finite) * 8.0, 1.0)


def _same_weekday_history(history: list[float], weeks: int = 8) -> list[float]:
    values: list[float] = []
    for offset in range(7, min(len(history), weeks * 7) + 1, 7):
        values.append(float(history[-offset]))
    return values


def _predict_one(history: list[float], model: str) -> float:
    if len(history) < 14:
        raise ValueError("At least 14 daily observations are required for adaptive forecasting")
    cap = _cap(history)
    if model == "seasonal_naive_7":
        value = history[-7]
    elif model == "moving_average_7":
        value = _mean(history[-7:])
    elif model == "moving_average_14":
        value = _mean(history[-14:])
    elif model == "median_7":
        value = _median(history[-7:])
    elif model == "median_14":
        value = _median(history[-14:])
    elif model == "weekday_mean_8w":
        value = _mean(_same_weekday_history(history, 8))
    elif model == "weekday_median_8w":
        value = _median(_same_weekday_history(history, 8))
    elif model == "seasonal_level_blend":
        seasonal = float(history[-7])
        recent_level = _median(history[-14:])
        value = 0.60 * seasonal + 0.40 * recent_level
    elif model == "weekly_trend_7":
        recent = sum(history[-7:])
        previous = sum(history[-14:-7])
        if previous <= 0:
            factor = 1.0
        else:
            raw = recent / previous
            factor = 1.0 + 0.55 * (min(1.35, max(0.65, raw)) - 1.0)
        value = history[-7] * min(1.20, max(0.80, factor))
    else:
        raise ValueError(f"Unsupported adaptive model: {model}")
    return min(max(0.0, float(value)), cap)


def _backtest(history: list[float], model: str) -> dict:
    start = max(14, len(history) - 42)
    actuals: list[float] = []
    predictions: list[float] = []
    residuals: list[float] = []
    for origin in range(start, len(history)):
        train = history[:origin]
        pred = _predict_one(train, model)
        actual = max(0.0, float(history[origin]))
        actuals.append(actual)
        predictions.append(pred)
        residuals.append(actual - pred)

    if not actuals:
        return {"wape": float("inf"), "mae": float("inf"), "rmse": float("inf"), "points": 0, "residuals": []}
    abs_error = sum(abs(a - p) for a, p in zip(actuals, predictions))
    denominator = sum(abs(a) for a in actuals)
    wape = abs_error / max(denominator, 1e-9)
    mae = abs_error / len(actuals)
    rmse = math.sqrt(sum((a - p) ** 2 for a, p in zip(actuals, predictions)) / len(actuals))
    return {"wape": wape, "mae": mae, "rmse": rmse, "points": len(actuals), "residuals": residuals}


def _select(history: list[float]) -> tuple[str, dict, dict[str, dict]]:
    results = {name: _backtest(history, name) for name in _CANDIDATES}
    selected = min(_CANDIDATES, key=lambda name: (results[name]["wape"], results[name]["mae"], name))
    return selected, results[selected], results


def forecast(history: list[float], last_date: date, horizon: int) -> list[dict]:
    if horizon not in {7, 30}:
        raise ValueError("Only 7-day and 30-day horizons are supported")
    if len(history) < 28:
        raise ValueError("At least 28 daily observations are required")

    clean = [max(0.0, float(v)) for v in history]
    selected, selected_metrics, candidates = _select(clean)
    residuals = [float(v) for v in selected_metrics["residuals"]]
    interval_error = max(_quantile([abs(v) for v in residuals], 0.90), 1.0)
    baseline = _mean(clean[-28:])

    mutable = list(clean)
    generated: list[dict] = []
    for offset in range(1, horizon + 1):
        target = last_date + timedelta(days=offset)
        predicted = _predict_one(mutable, selected)
        mutable.append(predicted)
        decline = max(0.0, (baseline - predicted) / max(abs(baseline), 1.0))
        generated.append({
            "date": target,
            "predicted": predicted,
            "lower": max(0.0, predicted - interval_error),
            "upper": predicted + interval_error,
            "baseline": baseline,
            "decline_probability": 0.0,
            "decline_percent": decline,
            "model_name": selected,
            "model_version": MODEL_VERSION,
            "metrics": {
                "mae": selected_metrics["mae"],
                "rmse": selected_metrics["rmse"],
                "wape": selected_metrics["wape"],
                "backtest_points": selected_metrics["points"],
                "selection_metric": "merchant_rolling_wape",
                "candidate_wape": {name: result["wape"] for name, result in candidates.items()},
                "evidence_scope": "uploaded_merchant_history",
            },
            "calibration": {
                "method": "merchant_local_rolling_model_selection_v2",
                "interval_method": "90pct_absolute_backtest_residual",
                "interval_error": interval_error,
            },
        })
    return generated
