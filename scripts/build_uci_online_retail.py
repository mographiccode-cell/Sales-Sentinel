from __future__ import annotations

import hashlib
import json
import math
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "online_retail.zip"
RAW = ROOT / "data" / "raw" / "Online Retail.xlsx"
DAILY = ROOT / "data" / "processed" / "daily_sales.csv"
MODEL = ROOT / "models" / "sales_forecast.json"
METRICS = ROOT / "reports" / "model_metrics.json"
MANIFEST = ROOT / "data" / "source_manifest.json"
URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
DOI = "10.24432/C5BW33"
FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_7", "rolling_14", "rolling_28", "std_7", "std_14",
    "weekday_sin", "weekday_cos", "month_sin", "month_cos", "trend",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> Path:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "Sales-Sentinel-Academic/1.0"})
        with urllib.request.urlopen(request, timeout=180) as source, ARCHIVE.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
    if not RAW.exists():
        with zipfile.ZipFile(ARCHIVE) as bundle:
            member = next(name for name in bundle.namelist() if name.lower().endswith(".xlsx"))
            with bundle.open(member) as source, RAW.open("wb") as target:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    return RAW


def clean(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(path)
    raw_rows = int(len(raw))
    duplicates = int(raw.duplicated().sum())
    frame = raw.drop_duplicates().copy()
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["UnitPrice"] = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    invalid = frame["InvoiceDate"].isna() | frame["Quantity"].isna() | frame["UnitPrice"].isna()
    invalid_rows = int(invalid.sum())
    frame = frame.loc[~invalid].copy()
    negative_price_rows = int((frame["UnitPrice"] < 0).sum())
    frame = frame.loc[frame["UnitPrice"] >= 0].copy()
    frame["is_return"] = frame["InvoiceNo"].astype(str).str.upper().str.startswith("C") | (frame["Quantity"] < 0)
    frame["amount"] = frame["Quantity"] * frame["UnitPrice"]
    frame["date"] = frame["InvoiceDate"].dt.normalize()

    grouped = frame.groupby("date", as_index=False).agg(
        gross_sales=("amount", lambda values: float(values[values > 0].sum())),
        returns=("amount", lambda values: float(abs(values[values < 0].sum()))),
        transactions=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
        quantity=("Quantity", lambda values: int(values[values > 0].sum())),
    )
    grouped["net_sales"] = grouped["gross_sales"] - grouped["returns"]
    calendar = pd.DataFrame({"date": pd.date_range(grouped.date.min(), grouped.date.max(), freq="D")})
    daily = calendar.merge(grouped, how="left", on="date")
    daily["observed_day"] = daily["net_sales"].notna().astype(int)
    numeric = ["gross_sales", "returns", "transactions", "customers", "quantity", "net_sales"]
    daily[numeric] = daily[numeric].fillna(0)
    daily["weekday"] = daily.date.dt.weekday
    daily["month"] = daily.date.dt.month

    audit = {
        "raw_rows": raw_rows,
        "duplicates_removed": duplicates,
        "invalid_rows_removed": invalid_rows,
        "negative_price_rows_removed": negative_price_rows,
        "clean_rows": int(len(frame)),
        "calendar_days": int(len(daily)),
        "observed_days": int(daily.observed_day.sum()),
        "zero_transaction_days": int((daily.observed_day == 0).sum()),
        "date_start": daily.date.min().date().isoformat(),
        "date_end": daily.date.max().date().isoformat(),
    }
    return daily, audit


def feature_row(history: list[float], target_date: date, trend: int) -> dict[str, float]:
    return {
        "lag_1": history[-1], "lag_2": history[-2], "lag_3": history[-3],
        "lag_7": history[-7], "lag_14": history[-14], "lag_21": history[-21], "lag_28": history[-28],
        "rolling_7": float(np.mean(history[-7:])),
        "rolling_14": float(np.mean(history[-14:])),
        "rolling_28": float(np.mean(history[-28:])),
        "std_7": float(np.std(history[-7:])),
        "std_14": float(np.std(history[-14:])),
        "weekday_sin": math.sin(2 * math.pi * target_date.weekday() / 7),
        "weekday_cos": math.cos(2 * math.pi * target_date.weekday() / 7),
        "month_sin": math.sin(2 * math.pi * (target_date.month - 1) / 12),
        "month_cos": math.cos(2 * math.pi * (target_date.month - 1) / 12),
        "trend": float(trend),
    }


def supervised(values: list[float], dates: list[date]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    targets: list[float] = []
    for index in range(28, len(values)):
        row = feature_row(values[:index], dates[index], index)
        rows.append([row[name] for name in FEATURES])
        targets.append(values[index])
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    if not np.isfinite(predicted).all():
        raise FloatingPointError("Model produced a non-finite prediction")
    error = actual - predicted
    denominator = max(float(np.abs(actual).sum()), 1e-9)
    smape_den = np.maximum(np.abs(actual) + np.abs(predicted), 1e-9)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(error).sum() / denominator),
        "smape": float(np.mean(2 * np.abs(error) / smape_den)),
    }


def prediction_cap(history: list[float]) -> float:
    values = np.asarray([max(0.0, value) for value in history if math.isfinite(value)], dtype=float)
    if not len(values):
        return 1.0
    return max(
        float(np.quantile(values, 0.995)) * 4.0,
        float(np.mean(values)) * 10.0,
        1.0,
    )


def baseline(history: list[float], horizon: int, name: str) -> np.ndarray:
    mutable = list(history)
    predictions: list[float] = []
    cap = prediction_cap(history)
    for _ in range(horizon):
        if name == "seasonal_naive_7":
            value = mutable[-7]
        elif name == "moving_average_7":
            value = float(np.mean(mutable[-7:]))
        elif name == "median_7":
            value = float(np.median(mutable[-7:]))
        elif name == "moving_average_14":
            value = float(np.mean(mutable[-14:]))
        else:
            raise ValueError(name)
        value = min(max(0.0, float(value)), cap)
        mutable.append(value)
        predictions.append(value)
    return np.asarray(predictions)


def fit_ridge(history: list[float], dates: list[date], alpha: float, transform: str):
    x, y = supervised(history, dates)
    scaler = StandardScaler().fit(x)
    target = np.log1p(np.maximum(y, 0)) if transform == "log1p" else y
    model = Ridge(alpha=alpha).fit(scaler.transform(x), target)
    return scaler, model


def ridge_forecast(history: list[float], dates: list[date], future_dates: list[date], alpha: float, transform: str):
    scaler, model = fit_ridge(history, dates, alpha, transform)
    mutable = list(history)
    predictions: list[float] = []
    cap = prediction_cap(history)
    max_log = math.log1p(cap)
    for target_date in future_dates:
        row = feature_row(mutable, target_date, len(mutable))
        vector = np.asarray([[row[name] for name in FEATURES]], dtype=float)
        if not np.isfinite(vector).all():
            raise FloatingPointError("Recursive features became non-finite")
        raw_value = float(model.predict(scaler.transform(vector))[0])
        if transform == "log1p":
            raw_value = min(max(raw_value, -20.0), max_log)
            value = math.expm1(raw_value)
        else:
            value = raw_value
        if not math.isfinite(value):
            raise FloatingPointError("Ridge model produced a non-finite prediction")
        value = min(max(0.0, value), cap)
        mutable.append(value)
        predictions.append(value)
    return np.asarray(predictions), scaler, model


def evaluate(values: list[float], dates: list[date], training_end: int, horizon: int) -> tuple[dict, dict]:
    history, history_dates = values[:training_end], dates[:training_end]
    actual = np.asarray(values[training_end: training_end + horizon], dtype=float)
    future_dates = dates[training_end: training_end + horizon]
    results: dict[str, dict] = {}
    predictions: dict[str, np.ndarray] = {}

    for name in ("seasonal_naive_7", "moving_average_7", "median_7", "moving_average_14"):
        prediction = baseline(history, horizon, name)
        predictions[name] = prediction
        results[name] = {**scores(actual, prediction), "status": "valid"}

    for alpha in (1.0, 10.0, 100.0, 1000.0):
        for transform in ("identity", "log1p"):
            name = f"ridge_{'log' if transform == 'log1p' else 'raw'}_{int(alpha)}"
            try:
                prediction, _, _ = ridge_forecast(history, history_dates, future_dates, alpha, transform)
                predictions[name] = prediction
                results[name] = {**scores(actual, prediction), "status": "valid"}
            except (FloatingPointError, OverflowError, ValueError) as exc:
                results[name] = {
                    "mae": 1e308,
                    "rmse": 1e308,
                    "wape": 1e308,
                    "smape": 2.0,
                    "status": "excluded_unstable",
                    "reason": str(exc),
                }
    return results, predictions


def train(daily: pd.DataFrame) -> tuple[dict, dict]:
    dates = [value.date() for value in daily.date]
    values = daily.net_sales.astype(float).tolist()
    if len(values) < 120:
        raise ValueError("At least 120 calendar days are required for chronological training")

    test_days = 30
    validation_days = 30
    validation_end = len(values) - test_days
    validation_start = validation_end - validation_days
    validation_metrics, _ = evaluate(values, dates, validation_start, validation_days)
    valid_candidates = {
        name: metrics for name, metrics in validation_metrics.items()
        if metrics.get("status") == "valid" and math.isfinite(metrics["wape"])
    }
    if not valid_candidates:
        raise RuntimeError("No numerically stable forecasting model was produced")
    selected = min(valid_candidates, key=lambda name: valid_candidates[name]["wape"])

    history = values[:validation_end]
    history_dates = dates[:validation_end]
    test_actual = np.asarray(values[validation_end:], dtype=float)
    test_dates = dates[validation_end:]
    if selected.startswith("ridge_"):
        parts = selected.split("_")
        transform = "log1p" if parts[1] == "log" else "identity"
        alpha = float(parts[2])
        test_prediction, _, _ = ridge_forecast(history, history_dates, test_dates, alpha, transform)
    else:
        transform = "identity"
        alpha = None
        test_prediction = baseline(history, test_days, selected)
    test_metrics = scores(test_actual, test_prediction)
    residuals = test_actual - test_prediction
    lower_error = float(np.quantile(residuals, 0.05))
    upper_error = float(np.quantile(residuals, 0.95))
    coverage = float(np.mean(
        (test_actual >= np.maximum(0, test_prediction + lower_error))
        & (test_actual <= test_prediction + upper_error)
    ))

    ridge_payload: dict = {}
    if selected.startswith("ridge_"):
        scaler, model = fit_ridge(values, dates, float(alpha), transform)
        ridge_payload = {
            "features": FEATURES,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
            "alpha": float(alpha),
            "target_transform": transform,
            "prediction_cap": prediction_cap(values),
        }

    artifact = {
        "version": "sales-sentinel-uci-online-retail-v2",
        "dataset": "UCI Online Retail",
        "dataset_doi": DOI,
        "selected_model": selected,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "history_start": dates[0].isoformat(),
        "history_end": dates[-1].isoformat(),
        "validation_days": validation_days,
        "test_days": test_days,
        "selection_metric": "validation_wape",
        "metrics": test_metrics,
        "validation_metrics": valid_candidates[selected],
        "all_validation_metrics": validation_metrics,
        "residual_quantiles": {
            "lower": lower_error,
            "upper": upper_error,
            "std": float(np.std(residuals)),
            "empirical_coverage": coverage,
        },
        "ridge": ridge_payload,
        "limitations": [
            "Historical retail behavior may differ from the deployment organization.",
            "Predictions are decision support, not guaranteed outcomes.",
            "Only 7-day and 30-day horizons are exposed by the application.",
        ],
    }
    report = {
        "selected_model": selected,
        "selection_metric": "validation_wape",
        "validation": validation_metrics,
        "test": {selected: test_metrics},
        "prediction_interval_90_coverage": coverage,
    }
    return artifact, report


def main() -> None:
    source = download()
    daily, audit = clean(source)
    artifact, report = train(daily)
    for path in (DAILY, MODEL, METRICS, MANIFEST):
        path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY, index=False, date_format="%Y-%m-%d")
    MODEL.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    METRICS.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {
        "dataset": "UCI Online Retail",
        "citation": "Chen, D. (2015). Online Retail. UCI Machine Learning Repository.",
        "doi": DOI,
        "license": "CC BY 4.0",
        "download_url": URL,
        "archive_sha256": sha256(ARCHIVE),
        "source_xlsx_sha256": sha256(source),
        "processed_sha256": sha256(DAILY),
        **audit,
        "synthetic_sales": False,
        "missing_calendar_days_are_zero_transaction_days": True,
        "pipeline": "scripts/build_uci_online_retail.py",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "success",
        "selected_model": artifact["selected_model"],
        "test_metrics": artifact["metrics"],
        "interval_coverage": artifact["residual_quantiles"]["empirical_coverage"],
        **audit,
    }, indent=2))


if __name__ == "__main__":
    main()
