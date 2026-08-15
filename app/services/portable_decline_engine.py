from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "models" / "sales_sentinel_portable_v18.json.gz"
SAMA_WEEKLY = ROOT / "data" / "saudi_v1_3" / "saudi_weekly_sama_calibration_v1_3.csv"
RICH_EXCLUDED_TYPES = {"DAILY_AGGREGATE", "DAILY_IMPORT"}


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    if not ARTIFACT.exists():
        raise RuntimeError("Sales Sentinel V18 artifact is missing")
    with gzip.open(ARTIFACT, "rt", encoding="utf-8") as stream:
        artifact = json.load(stream)
    required = {"version", "feature_names", "preprocessing", "trees", "decision_policy"}
    if not required.issubset(artifact):
        raise RuntimeError("Sales Sentinel V18 artifact is invalid")
    return artifact


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _safe_ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or not math.isfinite(float(a)) or not math.isfinite(float(b)) or float(b) == 0.0:
        return None
    return float(a) / float(b)


def _safe_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or not math.isfinite(float(current)) or not math.isfinite(float(previous)) or float(previous) == 0.0:
        return None
    return (float(current) - float(previous)) / abs(float(previous))


def _rolling(values: list[float], window: int) -> list[float] | None:
    if len(values) < window:
        return None
    return values[-window:]


def _prior(values: list[float], lag: int) -> float | None:
    return values[-1-lag] if len(values) > lag else None


@lru_cache(maxsize=1)
def _sama_lookup() -> dict[date, float]:
    result: dict[date, float] = {}
    if not SAMA_WEEKLY.exists():
        return result
    with SAMA_WEEKLY.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                result[date.fromisoformat(row["SAMAWeekStart"][:10])] = float(row["SAMAWeeklyMarketIndex"])
            except (ValueError, KeyError, TypeError):
                continue
    return result


def _week_start_sunday(day: date) -> date:
    return day - timedelta(days=(day.weekday() + 1) % 7)


def _market_series(days: list[date]) -> tuple[list[float | None], bool]:
    lookup = _sama_lookup()
    values: list[float | None] = []
    available = False
    for day in days:
        prior_week = _week_start_sunday(day) - timedelta(days=7)
        value = lookup.get(prior_week)
        values.append(value)
        available = available or value is not None
    return values, available


def _load_daily(db) -> list[dict]:
    excluded = tuple(sorted(RICH_EXCLUDED_TYPES))
    bounds = db.execute(text("""
        SELECT MIN(sale_date), MAX(sale_date)
        FROM sales
        WHERE transaction_type NOT IN (:t1, :t2)
    """), {"t1": excluded[0], "t2": excluded[1]}).one()
    if not bounds[0] or not bounds[1]:
        return []
    start = date.fromisoformat(str(bounds[0])[:10])
    end = date.fromisoformat(str(bounds[1])[:10])
    rows = db.execute(text("""
        SELECT
            sale_date,
            SUM(CAST(net_sales AS REAL)) AS sales,
            SUM(CASE WHEN CAST(gross_sales AS REAL) > 0 THEN CAST(gross_sales AS REAL) ELSE 0 END) AS gross_sales,
            COUNT(DISTINCT transaction_number) AS invoices,
            COUNT(DISTINCT CASE WHEN customer_key IS NOT NULL AND customer_key <> '' THEN customer_key END) AS customers,
            COUNT(DISTINCT product_id) AS products,
            SUM(ABS(quantity)) AS units,
            COUNT(*) AS transaction_rows,
            ABS(SUM(CASE WHEN CAST(net_sales AS REAL) < 0 THEN CAST(net_sales AS REAL) ELSE 0 END)) AS return_value
        FROM sales
        WHERE transaction_type NOT IN (:t1, :t2)
        GROUP BY sale_date
        ORDER BY sale_date
    """), {"t1": excluded[0], "t2": excluded[1]}).all()
    by_day = {date.fromisoformat(str(r[0])[:10]): r for r in rows}
    daily: list[dict] = []
    cursor = start
    while cursor <= end:
        r = by_day.get(cursor)
        if r:
            sales = float(r[1] or 0.0); gross = float(r[2] or 0.0); invoices = int(r[3] or 0)
            customers = int(r[4] or 0); products = int(r[5] or 0); units = float(r[6] or 0.0)
            transaction_rows = int(r[7] or 0); return_value = float(r[8] or 0.0)
        else:
            sales = gross = units = return_value = 0.0
            invoices = customers = products = transaction_rows = 0
        daily.append({
            "date": cursor,
            "sama_calibrated_net_sales_sar": sales,
            "gross_sales_sar": gross,
            "invoice_count": float(invoices),
            "unique_observed_customers": float(customers),
            "unique_products": float(products),
            "units": units,
            "average_invoice_value_sar": sales / max(invoices, 1),
            "return_rate_value": return_value / max(gross, 1e-9),
            "transaction_rows": float(transaction_rows),
        })
        cursor += timedelta(days=1)
    return daily


def _feature_map(daily: list[dict]) -> tuple[dict[str, float | None], bool]:
    feature: dict[str, float | None] = {}
    if not daily:
        return feature, False
    market, market_available = _market_series([row["date"] for row in daily])
    base_names = [
        "sama_calibrated_net_sales_sar", "gross_sales_sar", "invoice_count",
        "unique_observed_customers", "unique_products", "units",
        "average_invoice_value_sar", "return_rate_value", "transaction_rows",
    ]
    for name in base_names:
        series = [float(row[name]) for row in daily]
        current = series[-1]
        prefix = name.replace("sama_calibrated_", "")
        for window in (7, 14, 28, 56):
            values = _rolling(series, window)
            mean = _mean(values) if values else None
            std = _std(values) if values else None
            feature[f"merchant__{prefix}__ratio_mean_{window}"] = _safe_ratio(current, mean)
            if name in {"sama_calibrated_net_sales_sar", "invoice_count", "unique_observed_customers"}:
                feature[f"merchant__{prefix}__z_{window}"] = None if std in (None, 0.0) else (current - float(mean)) / float(std)
        for lag in (1, 7, 14, 28):
            feature[f"merchant__{prefix}__change_{lag}"] = _safe_change(current, _prior(series, lag))

    sales = [float(row["sama_calibrated_net_sales_sar"]) for row in daily]
    ma7 = _mean(_rolling(sales, 7) or [])
    ma14 = _mean(_rolling(sales, 14) or [])
    ma28 = _mean(_rolling(sales, 28) or [])
    ma56 = _mean(_rolling(sales, 56) or [])
    vol7 = _std(_rolling(sales, 7) or [])
    vol28 = _std(_rolling(sales, 28) or [])
    feature["merchant__sales__ma7_vs_ma28"] = _safe_ratio(ma7, ma28)
    feature["merchant__sales__ma14_vs_ma56"] = _safe_ratio(ma14, ma56)
    feature["merchant__sales__vol7_vs_vol28"] = _safe_ratio(vol7, vol28)
    last28 = _rolling(sales, 28)
    feature["merchant__sales__drawdown28"] = None if not last28 else (_safe_ratio(sales[-1], max(last28)) or 0.0) - 1.0

    current_market = market[-1] if market else None
    feature["market__sama_weekly_market_index"] = current_market
    feature["market__sama_weekly_market_index__change_1"] = _safe_change(current_market, market[-2] if len(market) >= 2 else None)
    feature["market__sama_weekly_market_index__change_7"] = _safe_change(current_market, market[-8] if len(market) >= 8 else None)

    d = daily[-1]["date"]
    dow = float(d.weekday())
    feature["calendar__dow_sin"] = math.sin(2 * math.pi * dow / 7.0)
    feature["calendar__dow_cos"] = math.cos(2 * math.pi * dow / 7.0)
    feature["calendar__salary_period"] = 1.0 if 24 <= d.day <= 31 else 0.0
    feature["calendar__national_day_window"] = 1.0 if d.month == 9 and 16 <= d.day <= 30 else 0.0
    feature["calendar__founding_day_window"] = 1.0 if d.month == 2 and 15 <= d.day <= 29 else 0.0
    return feature, market_available


def _prepare_row(artifact: dict, features: dict[str, float | None]) -> list[float]:
    row: list[float] = []
    for name in artifact["feature_names"]:
        meta = artifact["preprocessing"][name]
        value = features.get(name)
        if value is None or not math.isfinite(float(value)):
            value = float(meta["median"])
        value = min(max(float(value), float(meta["p01"])), float(meta["p99"]))
        row.append(value)
    return row


def _score(artifact: dict, row: list[float]) -> float:
    total = 0.0
    trees = artifact["trees"]
    for tree in trees:
        node = 0
        while int(tree["feature"][node]) >= 0:
            index = int(tree["feature"][node])
            node = int(tree["left"][node]) if row[index] <= float(tree["threshold"][node]) else int(tree["right"][node])
        total += float(tree["p1"][node])
    return total / len(trees)


def _prior_v18_scores(db, version: str, lookback: int) -> list[float]:
    rows = db.execute(text("""
        SELECT mr.id, MAX(f.decline_probability) AS risk
        FROM model_runs mr
        JOIN forecasts f ON f.model_run_id = mr.id
        WHERE mr.model_version = :version AND mr.status = 'completed'
        GROUP BY mr.id, mr.completed_at
        ORDER BY mr.completed_at DESC
        LIMIT :limit
    """), {"version": version, "limit": int(lookback)}).all()
    return [float(row[1]) for row in reversed(rows) if row[1] is not None]


def assess_decline_risk(db) -> dict:
    artifact = load_artifact()
    daily = _load_daily(db)
    required = int(artifact.get("history_required_days", 56))
    if len(daily) < required:
        return {"available": False, "reason": f"V18 requires at least {required} calendar days of transaction-level history; found {len(daily)}.", "history_days": len(daily)}
    customer_days = sum(1 for row in daily[-required:] if row["unique_observed_customers"] > 0)
    product_days = sum(1 for row in daily[-required:] if row["unique_products"] > 0)
    if customer_days < 7 or product_days < 7:
        return {"available": False, "reason": "V18 requires transaction-level customer and product identifiers; daily aggregate imports run in minimal mode.", "history_days": len(daily), "customer_days": customer_days, "product_days": product_days}

    features, market_available = _feature_map(daily)
    row = _prepare_row(artifact, features)
    score = _score(artifact, row)
    policy = artifact["decision_policy"]
    static_threshold = float(policy["static_threshold"])
    mode = "static"
    threshold = static_threshold
    prior = _prior_v18_scores(db, artifact["version"], int(policy["percentile_lookback"]))
    percentile = None
    if bool(policy.get("causal_percentile_enabled")) and len(prior) >= int(policy["percentile_warmup"]):
        less = sum(1 for value in prior if value < score)
        equal = sum(1 for value in prior if value == score)
        percentile = (less + 0.5 * equal) / len(prior)
        threshold = float(policy["percentile_cutoff"])
        alert = percentile >= threshold
        mode = "causal_percentile"
    else:
        alert = score >= static_threshold

    return {
        "available": True,
        "score": score,
        "alert": bool(alert),
        "policy_mode": mode,
        "decision_threshold": threshold,
        "risk_percentile": percentile,
        "model_name": "Sales Sentinel Portable ExtraTrees",
        "model_version": artifact["version"],
        "feature_count": len(artifact["feature_names"]),
        "tree_count": len(artifact["trees"]),
        "history_days": len(daily),
        "data_start": daily[0]["date"],
        "data_end": daily[-1]["date"],
        "market_context_available": market_available,
        "red_supported": False,
    }
