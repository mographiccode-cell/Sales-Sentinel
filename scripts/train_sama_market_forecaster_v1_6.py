from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
VALUE_FILE = ROOT / "data" / "sama_pos" / "sama_pos_national_weekly_value_2020_2025.csv"
COUNT_FILE = ROOT / "data" / "sama_pos" / "sama_pos_national_weekly_count_2020_2025.csv"
OUT_DIR = ROOT / "data" / "sama_pos"
MODEL_DIR = ROOT / "models" / "sama_market_v1_6"
REPORT_DIR = ROOT / "reports" / "sama_market_v1_6"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FORECAST_CSV = OUT_DIR / "sama_market_walkforward_forecasts_2023_2025.csv"
REPORT_JSON = REPORT_DIR / "sama_market_forecaster_report_v1_6.json"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("week_start").reset_index(drop=True)
    v = np.log1p(d["value_thousand_sar"].astype(float))
    c = np.log1p(d["transaction_count"].astype(float))
    x = pd.DataFrame({"week_start": d["week_start"]})
    for name, s in {"log_value": v, "log_count": c}.items():
        x[f"{name}_t0"] = s
        for lag in (1, 2, 3, 4, 8, 13, 26, 52):
            x[f"{name}_lag_{lag}"] = s.shift(lag)
        for w in (4, 8, 13, 26, 52):
            x[f"{name}_mean_{w}"] = s.rolling(w).mean()
            x[f"{name}_std_{w}"] = s.rolling(w).std()
    week = d["week_start"].dt.isocalendar().week.astype(float)
    x["week_sin"] = np.sin(2*np.pi*week/52.18)
    x["week_cos"] = np.cos(2*np.pi*week/52.18)
    x["trend"] = np.arange(len(x), dtype=float)
    x["value_h1"] = v.shift(-1)
    x["value_h2"] = v.shift(-2)
    x["count_h1"] = c.shift(-1)
    x["count_h2"] = c.shift(-2)
    return x


def make_model():
    return HistGradientBoostingRegressor(
        learning_rate=0.045,
        max_iter=300,
        max_leaf_nodes=12,
        min_samples_leaf=10,
        l2_regularization=2.0,
        random_state=SEED,
    )


def wape(y, p):
    return float(np.abs(y-p).sum()/max(np.abs(y).sum(), 1e-9))


def main():
    value = pd.read_csv(VALUE_FILE, parse_dates=["week_start", "week_end"])
    count = pd.read_csv(COUNT_FILE, parse_dates=["week_start", "week_end"])
    d = value[["week_start", "value_thousand_sar"]].merge(count[["week_start", "transaction_count"]], on="week_start", how="inner").sort_values("week_start").reset_index(drop=True)
    f = build_features(d)
    target_cols = ["value_h1", "value_h2", "count_h1", "count_h2"]
    feature_cols = [c for c in f.columns if c not in {"week_start", *target_cols}]
    valid_features = f[feature_cols].notna().all(axis=1)

    start = pd.Timestamp("2022-12-25")
    rows = []
    for idx in f.index[f["week_start"] >= start]:
        origin = f.loc[idx, "week_start"]
        if not valid_features.loc[idx]:
            continue
        preds = {}
        actuals = {}
        for target in target_cols:
            train_mask = (f.index < idx) & valid_features & f[target].notna()
            # target h2 of an origin two weeks ago may include the current origin week;
            # require the target week to be strictly <= origin week for training availability.
            horizon = 2 if target.endswith("h2") else 1
            train_mask &= (f["week_start"] + pd.to_timedelta(7*horizon, unit="D") <= origin)
            tr = f.loc[train_mask]
            if len(tr) < 80:
                preds[target] = np.nan
                actuals[target] = np.nan
                continue
            model = make_model().fit(tr[feature_cols], tr[target])
            pred_log = float(model.predict(f.loc[[idx], feature_cols])[0])
            preds[target] = float(np.expm1(pred_log))
            if pd.notna(f.loc[idx, target]):
                actuals[target] = float(np.expm1(f.loc[idx, target]))
            else:
                actuals[target] = np.nan

        hist = d[d["week_start"] <= origin].tail(52)
        value_scale = float(hist["value_thousand_sar"].median())
        count_scale = float(hist["transaction_count"].median())
        last_value = float(d.loc[idx, "value_thousand_sar"])
        last_count = float(d.loc[idx, "transaction_count"])
        rows.append({
            "origin_week_start": origin,
            "forecast_h1_week_start": origin + pd.Timedelta(days=7),
            "forecast_h2_week_start": origin + pd.Timedelta(days=14),
            "predicted_value_h1": preds["value_h1"],
            "predicted_value_h2": preds["value_h2"],
            "predicted_count_h1": preds["count_h1"],
            "predicted_count_h2": preds["count_h2"],
            "actual_value_h1": actuals["value_h1"],
            "actual_value_h2": actuals["value_h2"],
            "actual_count_h1": actuals["count_h1"],
            "actual_count_h2": actuals["count_h2"],
            "predicted_value_h1_index_52median": preds["value_h1"]/value_scale if pd.notna(preds["value_h1"]) else np.nan,
            "predicted_value_h2_index_52median": preds["value_h2"]/value_scale if pd.notna(preds["value_h2"]) else np.nan,
            "predicted_count_h1_index_52median": preds["count_h1"]/count_scale if pd.notna(preds["count_h1"]) else np.nan,
            "predicted_count_h2_index_52median": preds["count_h2"]/count_scale if pd.notna(preds["count_h2"]) else np.nan,
            "predicted_value_h1_change_vs_last": preds["value_h1"]/last_value - 1 if pd.notna(preds["value_h1"]) else np.nan,
            "predicted_value_h2_change_vs_last": preds["value_h2"]/last_value - 1 if pd.notna(preds["value_h2"]) else np.nan,
            "predicted_count_h1_change_vs_last": preds["count_h1"]/last_count - 1 if pd.notna(preds["count_h1"]) else np.nan,
            "predicted_count_h2_change_vs_last": preds["count_h2"]/last_count - 1 if pd.notna(preds["count_h2"]) else np.nan,
        })
    out = pd.DataFrame(rows)
    out.to_csv(FORECAST_CSV, index=False)

    eval_rows = out[(out["origin_week_start"] >= "2023-01-01") & out["actual_value_h2"].notna()].copy()
    metrics = {}
    for metric_name, actual_col, pred_col in [
        ("value_h1", "actual_value_h1", "predicted_value_h1"),
        ("value_h2", "actual_value_h2", "predicted_value_h2"),
        ("count_h1", "actual_count_h1", "predicted_count_h1"),
        ("count_h2", "actual_count_h2", "predicted_count_h2"),
    ]:
        e = eval_rows[[actual_col, pred_col]].dropna()
        metrics[metric_name] = {
            "rows": int(len(e)),
            "MAE": float(mean_absolute_error(e[actual_col], e[pred_col])),
            "WAPE": wape(e[actual_col].to_numpy(), e[pred_col].to_numpy()),
            "correlation": float(e[actual_col].corr(e[pred_col])),
        }

    # Fit deployment models on all rows with known target.
    deployment = {}
    for target in target_cols:
        tr = f[valid_features & f[target].notna()]
        model = make_model().fit(tr[feature_cols], tr[target])
        path = MODEL_DIR / f"sama_{target}_forecaster.joblib"
        joblib.dump({"model": model, "features": feature_cols, "target_log1p": True, "target": target}, path)
        deployment[target] = str(path.relative_to(ROOT))

    report = {
        "version": "SAMA-MARKET-FORECASTER-1.6",
        "source_weeks": int(len(d)),
        "source_start": str(d["week_start"].min().date()),
        "source_end": str(d["week_start"].max().date()),
        "walkforward_forecast_rows": int(len(out)),
        "evaluation_start": "2023-01-01",
        "metrics": metrics,
        "models": deployment,
        "leakage_controls": {
            "each_walkforward_prediction_trains_only_on_origins_strictly_before_current_origin": True,
            "horizon_targets_required_to_be_available_by_origin_week": True,
            "future_actual_sama_values_are_not_features": True,
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
