from __future__ import annotations

import math
from datetime import date, timedelta

MODEL_VERSION = "SALES-SENTINEL-ADAPTIVE-MERCHANT-FORECAST-V3"
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
    return [float(history[-offset]) for offset in range(7, min(len(history), weeks * 7) + 1, 7)]


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
        value = 0.60 * float(history[-7]) + 0.40 * _median(history[-14:])
    elif model == "weekly_trend_7":
        recent = sum(history[-7:])
        previous = sum(history[-14:-7])
        factor = 1.0 if previous <= 0 else 1.0 + 0.55 * (min(1.35, max(0.65, recent / previous)) - 1.0)
        value = history[-7] * min(1.20, max(0.80, factor))
    else:
        raise ValueError(f"Unsupported adaptive model: {model}")
    return min(max(0.0, float(value)), cap)


def _simulate(history: list[float], model: str, horizon: int) -> list[float]:
    mutable = list(history)
    out: list[float] = []
    for _ in range(horizon):
        pred = _predict_one(mutable, model)
        mutable.append(pred)
        out.append(pred)
    return out


def _one_step_backtest(history: list[float], model: str) -> dict:
    start = max(14, len(history) - 42)
    actuals: list[float] = []
    predictions: list[float] = []
    residuals: list[float] = []
    for origin in range(start, len(history)):
        pred = _predict_one(history[:origin], model)
        actual = max(0.0, float(history[origin]))
        actuals.append(actual)
        predictions.append(pred)
        residuals.append(actual - pred)
    if not actuals:
        return {"wape": float("inf"), "mae": float("inf"), "rmse": float("inf"), "points": 0, "residuals": []}
    abs_error = sum(abs(a - p) for a, p in zip(actuals, predictions))
    denom = sum(abs(a) for a in actuals)
    return {
        "wape": abs_error / max(denom, 1e-9),
        "mae": abs_error / len(actuals),
        "rmse": math.sqrt(sum((a - p) ** 2 for a, p in zip(actuals, predictions)) / len(actuals)),
        "points": len(actuals),
        "residuals": residuals,
    }


def _horizon_backtest(history: list[float], model: str, horizon: int) -> dict:
    earliest = 28
    latest = len(history) - horizon
    if latest < earliest:
        return {"total_wape": float("inf"), "folds": 0}
    start = max(earliest, latest - 56)
    step = 7
    actual_totals: list[float] = []
    predicted_totals: list[float] = []
    for origin in range(start, latest + 1, step):
        actual = history[origin: origin + horizon]
        predicted = _simulate(history[:origin], model, horizon)
        actual_totals.append(sum(max(0.0, float(v)) for v in actual))
        predicted_totals.append(sum(predicted))
    if not actual_totals:
        return {"total_wape": float("inf"), "folds": 0}
    error = sum(abs(a - p) for a, p in zip(actual_totals, predicted_totals))
    denom = sum(abs(a) for a in actual_totals)
    return {"total_wape": error / max(denom, 1e-9), "folds": len(actual_totals)}


def _select(history: list[float], horizon: int) -> tuple[str, dict, dict[str, dict]]:
    results: dict[str, dict] = {}
    for name in _CANDIDATES:
        one = _one_step_backtest(history, name)
        multi = _horizon_backtest(history, name, horizon)
        results[name] = {**one, **multi}
    def key(name: str):
        result = results[name]
        primary = result["total_wape"] if math.isfinite(result["total_wape"]) else result["wape"]
        return (primary, result["wape"], result["mae"], name)
    selected = min(_CANDIDATES, key=key)
    return selected, results[selected], results


def forecast(history: list[float], last_date: date, horizon: int) -> list[dict]:
    if horizon not in {7, 30}:
        raise ValueError("Only 7-day and 30-day horizons are supported")
    if len(history) < 28:
        raise ValueError("At least 28 daily observations are required")

    clean = [max(0.0, float(v)) for v in history]
    selected, selected_metrics, candidates = _select(clean, horizon)
    residuals = [float(v) for v in selected_metrics["residuals"]]
    interval_error = max(_quantile([abs(v) for v in residuals], 0.90), 1.0)
    baseline = _mean(clean[-28:])
    predicted_path = _simulate(clean, selected, horizon)

    generated: list[dict] = []
    for offset, predicted in enumerate(predicted_path, start=1):
        target = last_date + timedelta(days=offset)
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
                "horizon_total_wape": selected_metrics["total_wape"] if math.isfinite(selected_metrics["total_wape"]) else None,
                "horizon_backtest_folds": selected_metrics["folds"],
                "backtest_points": selected_metrics["points"],
                "selection_metric": "merchant_horizon_total_wape_then_daily_wape",
                "candidate_wape": {name: result["wape"] for name, result in candidates.items()},
                "candidate_horizon_total_wape": {
                    name: (result["total_wape"] if math.isfinite(result["total_wape"]) else None)
                    for name, result in candidates.items()
                },
                "evidence_scope": "uploaded_merchant_history",
            },
            "calibration": {
                "method": "merchant_horizon_aware_model_selection_v3",
                "interval_method": "90pct_absolute_one_step_backtest_residual",
                "interval_error": interval_error,
            },
        })
    return generated
