from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Online Retail.xlsx"
DAILY = ROOT / "data" / "processed" / "daily_sales.csv"
MODEL = ROOT / "models" / "sales_forecast.json"
METRICS = ROOT / "reports" / "model_metrics.json"
MANIFEST = ROOT / "data" / "source_manifest.json"
URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
DOI = "10.24432/C5BW33"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download() -> Path:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if RAW.exists():
        return RAW
    archive = RAW.with_suffix(".zip")
    urllib.request.urlretrieve(URL, archive)
    import zipfile
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.lower().endswith(".xlsx"))
        with bundle.open(member) as source, RAW.open("wb") as target:
            target.write(source.read())
    return RAW


def clean(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_excel(path)
    raw_rows = len(raw)
    duplicates = int(raw.duplicated().sum())
    frame = raw.drop_duplicates().copy()
    frame["InvoiceDate"] = pd.to_datetime(frame["InvoiceDate"], errors="coerce")
    frame["Quantity"] = pd.to_numeric(frame["Quantity"], errors="coerce")
    frame["UnitPrice"] = pd.to_numeric(frame["UnitPrice"], errors="coerce")
    invalid = frame["InvoiceDate"].isna() | frame["Quantity"].isna() | frame["UnitPrice"].isna()
    frame = frame.loc[~invalid].copy()
    frame["is_return"] = frame["InvoiceNo"].astype(str).str.upper().str.startswith("C") | (frame["Quantity"] < 0)
    frame = frame.loc[frame["UnitPrice"] >= 0].copy()
    frame["amount"] = frame["Quantity"] * frame["UnitPrice"]
    frame["date"] = frame["InvoiceDate"].dt.normalize()
    grouped = frame.groupby("date", as_index=False).agg(
        gross_sales=("amount", lambda s: float(s[s > 0].sum())),
        returns=("amount", lambda s: float(abs(s[s < 0].sum()))),
        transactions=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique"),
        quantity=("Quantity", lambda s: int(s[s > 0].sum())),
    )
    grouped["net_sales"] = grouped["gross_sales"] - grouped["returns"]
    calendar = pd.DataFrame({"date": pd.date_range(grouped.date.min(), grouped.date.max(), freq="D")})
    daily = calendar.merge(grouped, how="left", on="date")
    daily["observed_day"] = daily["net_sales"].notna().astype(int)
    for column in ["gross_sales", "returns", "transactions", "customers", "quantity", "net_sales"]:
        daily[column] = daily[column].fillna(0)
    daily["weekday"] = daily.date.dt.weekday
    daily["month"] = daily.date.dt.month
    audit = {
        "raw_rows": raw_rows,
        "duplicates_removed": duplicates,
        "invalid_rows_removed": int(invalid.sum()),
        "clean_rows": int(len(frame)),
        "calendar_days": int(len(daily)),
        "observed_days": int(daily.observed_day.sum()),
        "date_start": daily.date.min().date().isoformat(),
        "date_end": daily.date.max().date().isoformat(),
    }
    return daily, audit


def features(series: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame(index=series.index)
    for lag in (1, 2, 3, 7, 14, 28):
        result[f"lag_{lag}"] = series.shift(lag)
    result["rolling_7"] = series.shift(1).rolling(7).mean()
    result["rolling_28"] = series.shift(1).rolling(28).mean()
    result["weekday"] = series.index.weekday
    result["month"] = series.index.month
    result["trend"] = np.arange(len(series))
    return result


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = actual - predicted
    denominator = max(float(np.abs(actual).sum()), 1e-9)
    smape_den = np.maximum(np.abs(actual) + np.abs(predicted), 1e-9)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(error).sum() / denominator),
        "smape": float(np.mean(2 * np.abs(error) / smape_den)),
    }


def train(daily: pd.DataFrame) -> tuple[dict, dict]:
    series = daily.set_index("date")["net_sales"].astype(float)
    x = features(series)
    dataset = x.assign(target=series).dropna()
    holdout = min(56, max(28, len(dataset) // 5))
    train_set, test_set = dataset.iloc[:-holdout], dataset.iloc[-holdout:]
    actual = test_set.target.to_numpy()
    candidates: dict[str, tuple[np.ndarray, dict]] = {}
    candidates["seasonal_naive_7"] = (test_set.lag_7.to_numpy(), {})
    candidates["moving_average_7"] = (test_set.rolling_7.to_numpy(), {})
    columns = [column for column in train_set.columns if column != "target"]
    scaler = StandardScaler().fit(train_set[columns])
    ridge = Ridge(alpha=10.0).fit(scaler.transform(train_set[columns]), train_set.target)
    candidates["ridge_lag_calendar"] = (
        np.maximum(0, ridge.predict(scaler.transform(test_set[columns]))),
        {
            "features": columns,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coefficients": ridge.coef_.tolist(),
            "intercept": float(ridge.intercept_),
        },
    )
    results = {name: scores(actual, prediction) for name, (prediction, _) in candidates.items()}
    winner = min(results, key=lambda name: results[name]["wape"])
    prediction = candidates[winner][0]
    residuals = actual - prediction
    artifact = {
        "version": "uci-online-retail-v1",
        "selected_model": winner,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "history_end": series.index.max().date().isoformat(),
        "validation_days": holdout,
        "residual_quantiles": {
            "lower": float(np.quantile(residuals, 0.10)),
            "upper": float(np.quantile(residuals, 0.90)),
            "std": float(np.std(residuals)),
        },
        "ridge": candidates["ridge_lag_calendar"][1],
        "metrics": results[winner],
        "all_metrics": results,
    }
    return artifact, results


def main() -> None:
    source = download()
    daily, audit = clean(source)
    artifact, results = train(daily)
    for path in (DAILY, MODEL, METRICS, MANIFEST):
        path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(DAILY, index=False, date_format="%Y-%m-%d")
    MODEL.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    METRICS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    manifest = {
        "dataset": "UCI Online Retail",
        "doi": DOI,
        "license": "CC BY 4.0",
        "download_url": URL,
        "source_sha256": sha256(source),
        "processed_sha256": sha256(DAILY),
        **audit,
        "synthetic_sales": False,
        "pipeline": "scripts/build_uci_online_retail.py",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"selected_model": artifact["selected_model"], "metrics": artifact["metrics"], **audit}, indent=2))


if __name__ == "__main__":
    main()
