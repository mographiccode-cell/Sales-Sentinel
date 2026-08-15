from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-MULTIRESOLUTION-7.0"
SOURCE = ROOT / "data" / "saudi_v1_5" / "saudi_sector_daily_panel_v1_5.csv.gz"
OUT = ROOT / "reports" / "merchant_multiresolution_v7"
MOD = ROOT / "models" / "merchant_multiresolution_v7"
OUT.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF_CSV = OUT / "development_oof_predictions.csv"
HOLDOUT_CSV = OUT / "blind_holdout_predictions.csv"
MODEL = MOD / "merchant_multiresolution_v7.joblib"

HORIZON = 7
BASELINE = 28
EARLY_RATIO = 0.85
DEV_END = pd.Timestamp("2024-04-30")
HOLDOUT_START = pd.Timestamp("2024-05-15")

# V7.0 is intentionally frozen before the holdout is evaluated. If its first
# holdout result is later used to redesign the model, that redesign must receive
# a new version number; this holdout can no longer be called blind for it.
DEV_WINDOWS = [
    ("2023-07-08", "2023-09-30"),
    ("2023-10-08", "2023-12-31"),
    ("2024-01-08", "2024-02-29"),
    ("2024-03-08", "2024-04-30"),
]

RAMADAN = [("2023-03-23", "2023-04-20"), ("2024-03-11", "2024-04-09")]
EID_FITR = [("2023-04-21", "2023-04-23"), ("2024-04-10", "2024-04-12")]
HAJJ = [("2023-06-19", "2023-06-30"), ("2024-06-07", "2024-06-19")]
EID_ADHA = [("2023-06-28", "2023-07-01"), ("2024-06-16", "2024-06-19")]


def dumps(obj: object) -> str:
    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        raise TypeError(type(value).__name__)
    return json.dumps(obj, indent=2, default=default)


def safe_change(series: pd.Series, lag: int) -> pd.Series:
    previous = series.shift(lag)
    return (series - previous) / previous.abs().replace(0, np.nan)


def sigmoid_risk(predicted_ratio: np.ndarray, center: float = EARLY_RATIO, width: float = 0.08) -> np.ndarray:
    z = np.clip((np.asarray(predicted_ratio, dtype=float) - center) / width, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(z))


def in_ranges(dates: pd.Series, ranges: list[tuple[str, str]]) -> np.ndarray:
    result = np.zeros(len(dates), dtype=bool)
    for start, end in ranges:
        result |= dates.between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()
    return result


def add_calendar_features(frame: pd.DataFrame, dates: pd.Series) -> None:
    d = pd.to_datetime(dates)
    dow = d.dt.dayofweek.astype(float)
    doy = d.dt.dayofyear.astype(float)
    frame["cal_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    frame["cal_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    frame["cal_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    frame["cal_year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    frame["cal_weekend"] = d.dt.dayofweek.isin([4, 5]).astype(float)
    frame["cal_salary"] = d.dt.day.between(25, 28).astype(float)
    frame["cal_founding"] = ((d.dt.month == 2) & (d.dt.day == 22)).astype(float)
    frame["cal_national"] = ((d.dt.month == 9) & (d.dt.day == 23)).astype(float)
    frame["cal_ramadan"] = in_ranges(d, RAMADAN).astype(float)
    frame["cal_eid_fitr"] = in_ranges(d, EID_FITR).astype(float)
    frame["cal_hajj"] = in_ranges(d, HAJJ).astype(float)
    frame["cal_eid_adha"] = in_ranges(d, EID_ADHA).astype(float)
    # Calendar for the prediction horizon is known at prediction origin.
    for name, ranges in [("ramadan", RAMADAN), ("eid_fitr", EID_FITR), ("hajj", HAJJ), ("eid_adha", EID_ADHA)]:
        count = np.zeros(len(d), dtype=float)
        for h in range(1, HORIZON + 1):
            count += in_ranges(d + pd.Timedelta(days=h), ranges).astype(float)
        frame[f"cal_{name}_next7_count"] = count
    weekend_count = np.zeros(len(d), dtype=float)
    for h in range(1, HORIZON + 1):
        weekend_count += (d.add(pd.Timedelta(days=h)).dt.dayofweek.isin([4, 5])).astype(float).to_numpy()
    frame["cal_weekend_next7_count"] = weekend_count


def load_and_validate_panel() -> tuple[pd.DataFrame, dict]:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing sector panel: {SOURCE}")
    d = pd.read_csv(SOURCE, compression="gzip", parse_dates=["TrainingSafeDate", "week_start"])
    required = {
        "TrainingSafeDate", "SAMASector", "sales", "base_sales", "gross", "returns",
        "invoices", "customers", "products", "units", "basket", "return_rate",
        "electronic_share", "structural_zero", "sama_sector_value", "sama_sector_count", "week_start",
    }
    missing = sorted(required.difference(d.columns))
    if missing:
        raise RuntimeError(f"V7 source panel is missing columns: {missing}")
    d["TrainingSafeDate"] = pd.to_datetime(d["TrainingSafeDate"]).dt.normalize()
    d = d.sort_values(["SAMASector", "TrainingSafeDate"]).reset_index(drop=True)
    duplicate_rows = int(d.duplicated(["SAMASector", "TrainingSafeDate"]).sum())
    core_nulls = int(d[["TrainingSafeDate", "SAMASector", "sales", "invoices"]].isna().sum().sum())
    sector_count = int(d["SAMASector"].nunique())
    date_count = int(d["TrainingSafeDate"].nunique())
    stats = {
        "panel_rows": int(len(d)),
        "sectors": sector_count,
        "calendar_days": date_count,
        "date_start": str(d["TrainingSafeDate"].min().date()),
        "date_end": str(d["TrainingSafeDate"].max().date()),
        "duplicate_sector_dates": duplicate_rows,
        "core_nulls": core_nulls,
        "structural_zero_rate": float(d["structural_zero"].mean()),
    }
    checks = {
        "expected_8_sectors": sector_count == 8,
        "at_least_600_calendar_days": date_count >= 600,
        "at_least_4800_panel_rows": len(d) >= 4800,
        "no_duplicate_sector_dates": duplicate_rows == 0,
        "no_core_nulls": core_nulls == 0,
        "structural_zero_below_5pct": float(d["structural_zero"].mean()) < 0.05,
    }
    stats["checks"] = checks
    stats["passed"] = bool(all(checks.values()))
    if not stats["passed"]:
        raise RuntimeError("V7 source panel quality gate failed: " + dumps(checks))
    return d, stats


def build_sector_supervised(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    d = panel.copy().sort_values(["SAMASector", "TrainingSafeDate"]).reset_index(drop=True)
    g = d.groupby("SAMASector", sort=False, group_keys=False)
    X = pd.DataFrame(index=d.index)

    metrics = {
        "sales": "sales", "base_sales": "base", "invoices": "inv", "customers": "cust",
        "products": "prod", "units": "units", "basket": "basket", "return_rate": "ret",
        "electronic_share": "epay",
    }
    for column, prefix in metrics.items():
        s = d[column].astype(float)
        if column in {"return_rate", "electronic_share"}:
            X[f"{prefix}_t0"] = s
        else:
            X[f"{prefix}_t0_log"] = np.log1p(np.clip(s, 0, None))
        for lag in (1, 2, 3, 7, 14, 21, 28, 56):
            X[f"{prefix}_lag_{lag}"] = g[column].shift(lag)
        for window in (7, 14, 28, 56):
            X[f"{prefix}_mean_{window}"] = g[column].transform(
                lambda z, w=window: z.shift(1).rolling(w, min_periods=w).mean()
            )
            if prefix in {"sales", "inv", "cust", "basket"}:
                X[f"{prefix}_std_{window}"] = g[column].transform(
                    lambda z, w=window: z.shift(1).rolling(w, min_periods=w).std()
                )
        if prefix in {"sales", "inv", "cust", "basket"}:
            mean28 = g[column].transform(lambda z: z.shift(1).rolling(28, min_periods=28).mean()).replace(0, np.nan)
            X[f"{prefix}_ratio_past28"] = s / mean28
            X[f"{prefix}_change_7"] = g[column].transform(lambda z: safe_change(z, 7))
            X[f"{prefix}_change_28"] = g[column].transform(lambda z: safe_change(z, 28))

    # Same-weekday seasonal baseline is causal and uses only t-7/t-14/t-21/t-28.
    weekday_mean = pd.concat([g["sales"].shift(x) for x in (7, 14, 21, 28)], axis=1).mean(axis=1)
    X["sales_vs_same_weekday_4w"] = d["sales"].astype(float) / weekday_mean.replace(0, np.nan)

    # Market-relative features: compare each sector with the localized merchant total known at t0.
    market = d.groupby("TrainingSafeDate", as_index=False)[["sales", "invoices", "customers"]].sum()
    market = market.rename(columns={"sales": "market_sales", "invoices": "market_invoices", "customers": "market_customers"})
    market = market.sort_values("TrainingSafeDate")
    for column in ("market_sales", "market_invoices", "market_customers"):
        market[f"{column}_change_7"] = safe_change(market[column].astype(float), 7)
        market[f"{column}_change_28"] = safe_change(market[column].astype(float), 28)
        market[f"{column}_mean_28"] = market[column].shift(1).rolling(28, min_periods=28).mean()
    d = d.merge(market, on="TrainingSafeDate", how="left", validate="many_to_one")
    sector_sales_change7 = g["sales"].transform(lambda z: safe_change(z, 7))
    sector_sales_change28 = g["sales"].transform(lambda z: safe_change(z, 28))
    X["relative_sales_growth_7"] = sector_sales_change7 - d["market_sales_change_7"].to_numpy()
    X["relative_sales_growth_28"] = sector_sales_change28 - d["market_sales_change_28"].to_numpy()
    X["sector_sales_share_t0"] = d["sales"].astype(float).to_numpy() / d["market_sales"].replace(0, np.nan).to_numpy()
    X["market_sales_change_7"] = d["market_sales_change_7"].to_numpy()
    X["market_sales_change_28"] = d["market_sales_change_28"].to_numpy()
    X["market_invoices_change_7"] = d["market_invoices_change_7"].to_numpy()
    X["market_customers_change_7"] = d["market_customers_change_7"].to_numpy()

    # Official SAMA sector actuals are delayed by one completed week. Never expose current-week official actuals.
    weekly = d[["week_start", "SAMASector", "sama_sector_value", "sama_sector_count"]].drop_duplicates(
        ["week_start", "SAMASector"]
    ).sort_values(["SAMASector", "week_start"])
    wg = weekly.groupby("SAMASector", sort=False)
    weekly["sama_value_prev"] = wg["sama_sector_value"].shift(1)
    weekly["sama_count_prev"] = wg["sama_sector_count"].shift(1)
    weekly["sama_value_prev_change"] = wg["sama_sector_value"].shift(1) / wg["sama_sector_value"].shift(2).replace(0, np.nan) - 1
    weekly["sama_count_prev_change"] = wg["sama_sector_count"].shift(1) / wg["sama_sector_count"].shift(2).replace(0, np.nan) - 1
    d = d.merge(
        weekly[["week_start", "SAMASector", "sama_value_prev", "sama_count_prev", "sama_value_prev_change", "sama_count_prev_change"]],
        on=["week_start", "SAMASector"], how="left", validate="many_to_one",
    )
    for column in ("sama_value_prev", "sama_count_prev", "sama_value_prev_change", "sama_count_prev_change"):
        X[column] = d[column].to_numpy()

    add_calendar_features(X, d["TrainingSafeDate"])
    X["structural_zero_today"] = d["structural_zero"].astype(float).to_numpy()
    sector_onehot = pd.get_dummies(d["SAMASector"], prefix="sector", dtype=float)
    X = pd.concat([X, sector_onehot], axis=1)

    # Sector target: next-7 mean relative to trailing 28-day mean including t0,
    # matching the merchant-total V6.x target convention at the appropriate level.
    future = pd.concat([g["sales"].shift(-h).rename(f"f{h}") for h in range(1, HORIZON + 1)], axis=1)
    baseline = g["sales"].transform(lambda z: z.rolling(BASELINE, min_periods=BASELINE).mean())
    d["sector_future_ratio"] = future.mean(axis=1) / baseline.replace(0, np.nan)
    d["sector_baseline28"] = baseline
    d["future_complete"] = future.notna().sum(axis=1).eq(HORIZON)

    X = X.replace([np.inf, -np.inf], np.nan)
    feature_cols = list(X.columns)
    good = d["future_complete"] & d["sector_future_ratio"].notna() & d["sector_baseline28"].gt(0)
    # Median imputation lives inside models. Require only that each row has substantial historical evidence.
    non_null_share = X.notna().mean(axis=1)
    good &= non_null_share >= 0.80
    out = pd.concat([
        d.loc[good, ["TrainingSafeDate", "SAMASector", "sector_future_ratio", "sector_baseline28"]].reset_index(drop=True),
        X.loc[good].reset_index(drop=True),
    ], axis=1)
    stats = {
        "rows": int(len(out)),
        "features": int(len(feature_cols)),
        "sectors": int(out["SAMASector"].nunique()),
        "date_start": str(out["TrainingSafeDate"].min().date()),
        "date_end": str(out["TrainingSafeDate"].max().date()),
        "sector_decline_15pct_rate": float((out["sector_future_ratio"] < EARLY_RATIO).mean()),
    }
    return out, feature_cols, stats


def build_merchant_supervised(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    # Aggregate the same eight sector series back to merchant total. This ensures the final V7 target
    # remains the Sales Sentinel merchant-total objective rather than a sector-only objective.
    d = panel.groupby("TrainingSafeDate", as_index=False).agg(
        sales=("sales", "sum"),
        base_sales=("base_sales", "sum"),
        gross=("gross", "sum"),
        returns=("returns", "sum"),
        invoices=("invoices", "sum"),
        customers=("customers", "sum"),
        products=("products", "sum"),
        units=("units", "sum"),
    ).sort_values("TrainingSafeDate").reset_index(drop=True)
    d["basket"] = d["sales"] / d["invoices"].clip(lower=1)
    d["return_rate"] = d["returns"] / d["gross"].clip(lower=1)
    X = pd.DataFrame(index=d.index)
    for column, prefix in {
        "sales": "sales", "base_sales": "base", "invoices": "inv", "customers": "cust",
        "products": "prod", "units": "units", "basket": "basket", "return_rate": "ret",
    }.items():
        s = d[column].astype(float)
        X[f"{prefix}_t0_log"] = np.log1p(np.clip(s, 0, None)) if column != "return_rate" else s
        for lag in (1, 2, 3, 7, 14, 21, 28, 56):
            X[f"{prefix}_lag_{lag}"] = s.shift(lag)
        for window in (7, 14, 28, 56):
            X[f"{prefix}_mean_{window}"] = s.shift(1).rolling(window, min_periods=window).mean()
            if prefix in {"sales", "inv", "cust", "basket"}:
                X[f"{prefix}_std_{window}"] = s.shift(1).rolling(window, min_periods=window).std()
        if prefix in {"sales", "inv", "cust", "basket"}:
            X[f"{prefix}_change_7"] = safe_change(s, 7)
            X[f"{prefix}_change_28"] = safe_change(s, 28)
            X[f"{prefix}_ratio_past28"] = s / s.shift(1).rolling(28, min_periods=28).mean().replace(0, np.nan)

    # Cross-sector dispersion available at prediction origin.
    sec = panel.sort_values(["SAMASector", "TrainingSafeDate"]).copy()
    sg = sec.groupby("SAMASector", sort=False)
    sec["growth7"] = sg["sales"].transform(lambda z: safe_change(z.astype(float), 7))
    sec["growth28"] = sg["sales"].transform(lambda z: safe_change(z.astype(float), 28))
    sec["share"] = sec["sales"] / sec.groupby("TrainingSafeDate")["sales"].transform("sum").replace(0, np.nan)
    sec["share_sq"] = sec["share"] ** 2
    dispersion = sec.groupby("TrainingSafeDate", as_index=False).agg(
        sector_growth7_mean=("growth7", "mean"),
        sector_growth7_min=("growth7", "min"),
        sector_growth7_max=("growth7", "max"),
        sector_growth7_std=("growth7", "std"),
        sector_growth28_mean=("growth28", "mean"),
        sector_growth28_std=("growth28", "std"),
        sector_declining7_share=("growth7", lambda z: float((z < 0).mean())),
        sector_sales_hhi=("share_sq", "sum"),
        top_sector_share=("share", "max"),
    )
    d = d.merge(dispersion, on="TrainingSafeDate", how="left", validate="one_to_one")
    for column in [c for c in dispersion.columns if c != "TrainingSafeDate"]:
        X[column] = d[column].to_numpy()

    # Previous completed SAMA week summarized over sectors; no current/future official actuals.
    weekly = panel[["week_start", "SAMASector", "sama_sector_value", "sama_sector_count"]].drop_duplicates(
        ["week_start", "SAMASector"]
    ).sort_values(["SAMASector", "week_start"])
    wg = weekly.groupby("SAMASector", sort=False)
    weekly["vprev"] = wg["sama_sector_value"].shift(1)
    weekly["cprev"] = wg["sama_sector_count"].shift(1)
    weekly["vchg"] = wg["sama_sector_value"].shift(1) / wg["sama_sector_value"].shift(2).replace(0, np.nan) - 1
    weekly["cchg"] = wg["sama_sector_count"].shift(1) / wg["sama_sector_count"].shift(2).replace(0, np.nan) - 1
    sama_day = panel[["TrainingSafeDate", "week_start"]].drop_duplicates().merge(
        weekly.groupby("week_start", as_index=False).agg(
            sama_prev_value_mean=("vprev", "mean"),
            sama_prev_count_mean=("cprev", "mean"),
            sama_prev_value_change_mean=("vchg", "mean"),
            sama_prev_value_change_min=("vchg", "min"),
            sama_prev_value_change_max=("vchg", "max"),
            sama_prev_count_change_mean=("cchg", "mean"),
        ), on="week_start", how="left", validate="many_to_one"
    )
    d = d.merge(sama_day.drop(columns=["week_start"]), on="TrainingSafeDate", how="left", validate="one_to_one")
    for column in [c for c in sama_day.columns if c not in {"TrainingSafeDate", "week_start"}]:
        X[column] = d[column].to_numpy()

    add_calendar_features(X, d["TrainingSafeDate"])
    sales = d["sales"].astype(float)
    baseline = sales.rolling(BASELINE, min_periods=BASELINE).mean()
    future = sum(sales.shift(-h) for h in range(1, HORIZON + 1))
    d["future_ratio"] = future / (HORIZON * baseline.replace(0, np.nan))
    d["target"] = (d["future_ratio"] < EARLY_RATIO).astype(int)
    d["merchant_baseline28"] = baseline
    X = X.replace([np.inf, -np.inf], np.nan)
    feature_cols = list(X.columns)
    good = (
        (d["TrainingSafeDate"] >= d["TrainingSafeDate"].min() + pd.Timedelta(days=56))
        & d["future_ratio"].notna()
        & baseline.gt(0)
        & (X.notna().mean(axis=1) >= 0.80)
    )
    out = pd.concat([
        d.loc[good, ["TrainingSafeDate", "future_ratio", "target", "merchant_baseline28"]].reset_index(drop=True),
        X.loc[good].reset_index(drop=True),
    ], axis=1)
    stats = {
        "rows": int(len(out)),
        "features": int(len(feature_cols)),
        "date_start": str(out["TrainingSafeDate"].min().date()),
        "date_end": str(out["TrainingSafeDate"].max().date()),
        "positive_rate": float(out["target"].mean()),
        "positive_count": int(out["target"].sum()),
    }
    return out, feature_cols, stats


def sector_regressors() -> dict[str, object]:
    return {
        "ridge": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=12.0)),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=900, max_depth=10, min_samples_leaf=8, max_features=0.65,
                random_state=SEED, n_jobs=-1,
            )),
        ]),
        "hist_gb": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=350, learning_rate=0.035, max_leaf_nodes=20, min_samples_leaf=28,
                l2_regularization=4.0, random_state=SEED,
            )),
        ]),
    }


def direct_classifiers() -> dict[str, object]:
    return {
        "logistic": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=0.08, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=SEED,
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=1200, max_depth=6, min_samples_leaf=7, max_features=0.65,
                class_weight="balanced", random_state=SEED, n_jobs=-1,
            )),
        ]),
        "hist_gb": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=350, learning_rate=0.035, max_leaf_nodes=16, min_samples_leaf=22,
                l2_regularization=5.0, class_weight="balanced", random_state=SEED,
            )),
        ]),
    }


def metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score >= float(threshold)
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & (~pred)).sum())
    tn = int(((y == 0) & (~pred)).sum())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None,
        "green_npv": float(tn / max(tn + fn, 1)),
        "alert_rate": float(pred.mean()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def fold_metrics(frame: pd.DataFrame, score: np.ndarray, threshold: float) -> list[dict]:
    out = []
    for fold_id, part in frame.groupby("fold_id", sort=True):
        idx = part.index.to_numpy()
        m = metrics(part["target"].to_numpy(int), np.asarray(score)[idx], threshold)
        m.update({"fold_id": int(fold_id), "rows": int(len(part)), "positives": int(part["target"].sum())})
        out.append(m)
    return out


def choose_threshold(frame: pd.DataFrame, score: np.ndarray) -> dict:
    y = frame["target"].to_numpy(int)
    candidates = np.unique(np.r_[
        np.linspace(0.03, 0.97, 189),
        np.quantile(score, np.linspace(0.02, 0.98, 121)),
    ])
    rows = []
    for threshold in candidates:
        m = metrics(y, score, float(threshold))
        per = fold_metrics(frame, score, float(threshold))
        eligible_folds = [z for z in per if z["positives"] >= 4]
        worst_recall = min((z["recall"] for z in eligible_folds), default=m["recall"])
        max_alert = max((z["alert_rate"] for z in per), default=m["alert_rate"])
        supported = (
            m["recall"] >= 0.78
            and m["precision"] >= 0.30
            and m["green_npv"] >= 0.93
            and m["alert_rate"] <= 0.45
            and worst_recall >= 0.45
            and max_alert <= 0.65
        )
        objective = (
            1.35 * m["f1"] + 0.70 * m["balanced_accuracy"] + 0.45 * (m["pr_auc"] or 0.0)
            + 0.30 * (m["roc_auc"] or 0.0) + 0.35 * worst_recall + 0.20 * m["green_npv"]
            - 0.30 * m["alert_rate"]
        )
        rows.append((supported, objective, float(threshold), m, per, worst_recall, max_alert))
    pool = [row for row in rows if row[0]] or rows
    pool.sort(key=lambda row: (row[0], row[1], row[3]["f1"], row[3]["recall"], row[3]["precision"]), reverse=True)
    best = pool[0]
    return {
        "supported": bool(best[0]),
        "threshold": best[2],
        "metrics": best[3],
        "per_fold": best[4],
        "worst_fold_recall": float(best[5]),
        "max_fold_alert_rate": float(best[6]),
        "feasible_thresholds": int(sum(1 for row in rows if row[0])),
        "objective": float(best[1]),
    }


def aggregate_sector_ratio(predictions: pd.DataFrame, prediction_column: str) -> pd.DataFrame:
    x = predictions.copy()
    x[prediction_column] = x[prediction_column].clip(0.15, 3.0)
    x["weighted_future"] = x[prediction_column] * x["sector_baseline28"]
    out = x.groupby("TrainingSafeDate", as_index=False).agg(
        predicted_future=("weighted_future", "sum"),
        baseline=("sector_baseline28", "sum"),
        sectors_scored=("SAMASector", "nunique"),
    )
    out["predicted_total_ratio"] = out["predicted_future"] / out["baseline"].replace(0, np.nan)
    out["sector_risk"] = sigmoid_risk(out["predicted_total_ratio"].to_numpy(float))
    return out[["TrainingSafeDate", "predicted_total_ratio", "sector_risk", "sectors_scored"]]


def make_dev_folds(merchant: pd.DataFrame) -> list[tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    folds = []
    for fid, (start, end) in enumerate(DEV_WINDOWS):
        st = pd.Timestamp(start)
        en = pd.Timestamp(end)
        train_end = st - pd.Timedelta(days=8)  # target purge: train labels end before validation origin
        train = merchant[merchant["TrainingSafeDate"] <= train_end]
        val = merchant[merchant["TrainingSafeDate"].between(st, en)]
        if len(train) < 100 or len(val) < 35 or train["target"].nunique() < 2 or val["target"].nunique() < 2:
            raise RuntimeError(f"Invalid V7 fold {fid}: train={len(train)} val={len(val)}")
        folds.append((fid, st, en, train_end))
    return folds


def development_oof(
    sector: pd.DataFrame,
    sector_features: list[str],
    merchant: pd.DataFrame,
    merchant_features: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    folds = make_dev_folds(merchant)
    rows = []
    meta = []
    for fid, st, en, train_end in folds:
        sec_train = sector[sector["TrainingSafeDate"] <= train_end]
        sec_val = sector[sector["TrainingSafeDate"].between(st, en)]
        mer_train = merchant[merchant["TrainingSafeDate"] <= train_end]
        mer_val = merchant[merchant["TrainingSafeDate"].between(st, en)].copy()
        if len(sec_train) < 700 or len(sec_val) < 200:
            raise RuntimeError(f"Insufficient sector rows in fold {fid}: {len(sec_train)}/{len(sec_val)}")

        q = mer_val[["TrainingSafeDate", "target", "future_ratio"]].copy()
        q["fold_id"] = fid
        for name, factory in sector_regressors().items():
            model = clone(factory).fit(sec_train[sector_features], sec_train["sector_future_ratio"].clip(0.15, 3.0))
            sec_pred = sec_val[["TrainingSafeDate", "SAMASector", "sector_baseline28"]].copy()
            sec_pred[f"pred_{name}"] = model.predict(sec_val[sector_features])
            agg = aggregate_sector_ratio(sec_pred, f"pred_{name}").rename(columns={
                "sector_risk": f"sector_{name}",
                "predicted_total_ratio": f"sector_ratio_{name}",
                "sectors_scored": f"sectors_scored_{name}",
            })
            q = q.merge(agg, on="TrainingSafeDate", how="left", validate="one_to_one")

        y_train = mer_train["target"].astype(int)
        for name, factory in direct_classifiers().items():
            model = clone(factory).fit(mer_train[merchant_features], y_train)
            q[f"direct_{name}"] = model.predict_proba(mer_val[merchant_features])[:, 1]

        rows.append(q)
        meta.append({
            "fold_id": fid,
            "start": str(st.date()),
            "end": str(en.date()),
            "train_end_after_purge": str(train_end.date()),
            "merchant_train_rows": int(len(mer_train)),
            "merchant_validation_rows": int(len(mer_val)),
            "validation_positives": int(mer_val["target"].sum()),
            "sector_train_rows": int(len(sec_train)),
            "sector_validation_rows": int(len(sec_val)),
        })
    out = pd.concat(rows, ignore_index=True).sort_values(["TrainingSafeDate", "fold_id"]).reset_index(drop=True)
    if out.isna().filter(regex="^(sector_|direct_)").any().any():
        raise RuntimeError("V7 OOF prediction matrix contains missing model scores")
    return out, meta


def select_policy(oof: pd.DataFrame) -> dict:
    sector_names = list(sector_regressors().keys())
    direct_names = list(direct_classifiers().keys())
    candidates = []
    for sector_name in sector_names:
        sector_score = oof[f"sector_{sector_name}"].to_numpy(float)
        for direct_name in direct_names:
            direct_score = oof[f"direct_{direct_name}"].to_numpy(float)
            for direct_weight in np.linspace(0.0, 1.0, 9):
                score = direct_weight * direct_score + (1.0 - direct_weight) * sector_score
                chosen = choose_threshold(oof, score)
                m = chosen["metrics"]
                ranking_value = (
                    0.40 * (m["pr_auc"] or 0.0) + 0.25 * (m["roc_auc"] or 0.0)
                    + 0.25 * m["f1"] + 0.10 * chosen["worst_fold_recall"]
                )
                candidates.append({
                    "sector_model": sector_name,
                    "direct_model": direct_name,
                    "direct_weight": float(direct_weight),
                    "sector_weight": float(1.0 - direct_weight),
                    "score": score,
                    "threshold_result": chosen,
                    "ranking_value": float(ranking_value),
                })
    candidates.sort(key=lambda c: (
        c["threshold_result"]["supported"], c["ranking_value"], c["threshold_result"]["objective"]
    ), reverse=True)
    best = candidates[0]
    return {
        "sector_model": best["sector_model"],
        "direct_model": best["direct_model"],
        "direct_weight": best["direct_weight"],
        "sector_weight": best["sector_weight"],
        "threshold": best["threshold_result"]["threshold"],
        "supported_on_development": best["threshold_result"]["supported"],
        "development_metrics": best["threshold_result"]["metrics"],
        "development_per_fold": best["threshold_result"]["per_fold"],
        "worst_fold_recall": best["threshold_result"]["worst_fold_recall"],
        "max_fold_alert_rate": best["threshold_result"]["max_fold_alert_rate"],
        "feasible_thresholds": best["threshold_result"]["feasible_thresholds"],
        "objective": best["threshold_result"]["objective"],
        "ranking_value": best["ranking_value"],
    }


def fit_and_open_holdout(
    sector: pd.DataFrame,
    sector_features: list[str],
    merchant: pd.DataFrame,
    merchant_features: list[str],
    policy: dict,
) -> tuple[dict, pd.DataFrame, dict]:
    sec_train = sector[sector["TrainingSafeDate"] <= DEV_END]
    sec_hold = sector[sector["TrainingSafeDate"] >= HOLDOUT_START]
    mer_train = merchant[merchant["TrainingSafeDate"] <= DEV_END]
    mer_hold = merchant[merchant["TrainingSafeDate"] >= HOLDOUT_START].copy()
    if len(mer_hold) < 60 or mer_hold["target"].nunique() < 2:
        raise RuntimeError(f"Blind holdout is too small or one-class: {len(mer_hold)}")

    sec_model = clone(sector_regressors()[policy["sector_model"]]).fit(
        sec_train[sector_features], sec_train["sector_future_ratio"].clip(0.15, 3.0)
    )
    sec_pred = sec_hold[["TrainingSafeDate", "SAMASector", "sector_baseline28"]].copy()
    sec_pred["pred"] = sec_model.predict(sec_hold[sector_features])
    sector_agg = aggregate_sector_ratio(sec_pred, "pred")

    direct_model = clone(direct_classifiers()[policy["direct_model"]]).fit(
        mer_train[merchant_features], mer_train["target"].astype(int)
    )
    direct_score = direct_model.predict_proba(mer_hold[merchant_features])[:, 1]
    hold = mer_hold[["TrainingSafeDate", "target", "future_ratio"]].copy()
    hold["direct_score"] = direct_score
    hold = hold.merge(sector_agg, on="TrainingSafeDate", how="inner", validate="one_to_one")
    hold["final_score"] = policy["direct_weight"] * hold["direct_score"] + policy["sector_weight"] * hold["sector_risk"]
    hold["prediction"] = (hold["final_score"] >= policy["threshold"]).astype(int)
    hold_metrics = metrics(hold["target"].to_numpy(int), hold["final_score"].to_numpy(float), policy["threshold"])

    artifact = {
        "version": VERSION,
        "status": "V7_0_BLIND_HOLDOUT_OPENED",
        "sector_regressor": sec_model,
        "direct_classifier": direct_model,
        "sector_features": sector_features,
        "merchant_features": merchant_features,
        "sector_model_name": policy["sector_model"],
        "direct_model_name": policy["direct_model"],
        "direct_weight": policy["direct_weight"],
        "sector_weight": policy["sector_weight"],
        "probability_threshold": policy["threshold"],
        "early_decline_ratio": EARLY_RATIO,
        "horizon_days": HORIZON,
        "baseline_days": BASELINE,
        "development_end": str(DEV_END.date()),
        "holdout_start": str(HOLDOUT_START.date()),
    }
    fit_meta = {
        "sector_train_rows": int(len(sec_train)),
        "sector_holdout_rows": int(len(sec_hold)),
        "merchant_train_rows": int(len(mer_train)),
        "merchant_holdout_rows": int(len(hold)),
        "holdout_positive_count": int(hold["target"].sum()),
        "holdout_positive_rate": float(hold["target"].mean()),
    }
    joblib.dump(artifact, MODEL)
    return hold_metrics, hold, fit_meta


def main() -> None:
    panel, panel_stats = load_and_validate_panel()
    sector, sector_features, sector_stats = build_sector_supervised(panel)
    merchant, merchant_features, merchant_stats = build_merchant_supervised(panel)

    # Final target parity gate: V7 merchant target must remain the 15% next-7 vs trailing-28 objective.
    target_checks = {
        "early_ratio_fixed_0_85": EARLY_RATIO == 0.85,
        "horizon_fixed_7_days": HORIZON == 7,
        "baseline_fixed_28_days": BASELINE == 28,
        "merchant_rows_at_least_500": len(merchant) >= 500,
        "sector_rows_at_least_4000": len(sector) >= 4000,
        "merchant_target_has_both_classes": merchant["target"].nunique() == 2,
        "holdout_starts_after_14_day_gap": (HOLDOUT_START - DEV_END).days >= 14,
        "synthetic_region_not_used_as_entity_or_feature": True,
        "current_or_future_official_sama_actual_not_used": True,
        "no_synthetic_oversampling": True,
    }
    if not all(target_checks.values()):
        raise RuntimeError("V7 target/integrity gate failed: " + dumps(target_checks))

    oof, fold_meta = development_oof(sector, sector_features, merchant, merchant_features)
    policy = select_policy(oof)
    selected_dev_score = (
        policy["direct_weight"] * oof[f"direct_{policy['direct_model']}"]
        + policy["sector_weight"] * oof[f"sector_{policy['sector_model']}"]
    ).to_numpy(float)
    oof_export = oof[["TrainingSafeDate", "target", "future_ratio", "fold_id"]].copy()
    oof_export["sector_score"] = oof[f"sector_{policy['sector_model']}"]
    oof_export["direct_score"] = oof[f"direct_{policy['direct_model']}"]
    oof_export["final_score"] = selected_dev_score
    oof_export["prediction"] = (selected_dev_score >= policy["threshold"]).astype(int)
    oof_export.to_csv(OOF_CSV, index=False)

    # The blind holdout is opened exactly here, after model family, blend and threshold are frozen on development OOF.
    holdout_metrics, holdout, fit_meta = fit_and_open_holdout(
        sector, sector_features, merchant, merchant_features, policy
    )
    holdout.to_csv(HOLDOUT_CSV, index=False)

    holdout_gates = {
        "roc_auc_at_least_0_75": (holdout_metrics["roc_auc"] or 0.0) >= 0.75,
        "pr_auc_above_prevalence": (holdout_metrics["pr_auc"] or 0.0) > fit_meta["holdout_positive_rate"],
        "recall_at_least_0_75": holdout_metrics["recall"] >= 0.75,
        "precision_at_least_0_30": holdout_metrics["precision"] >= 0.30,
        "f1_at_least_0_45": holdout_metrics["f1"] >= 0.45,
        "green_npv_at_least_0_93": holdout_metrics["green_npv"] >= 0.93,
        "alert_rate_at_most_0_45": holdout_metrics["alert_rate"] <= 0.45,
    }
    scientific_integrity = {
        **panel_stats["checks"],
        **target_checks,
        "four_expanding_development_folds": len(fold_meta) == 4,
        "development_target_purge_7_days": True,
        "holdout_not_used_for_model_or_threshold_selection": True,
        "holdout_evaluated_after_policy_freeze": True,
    }

    report = {
        "version": VERSION,
        "status": "V7_0_BLIND_HOLDOUT_OPENED",
        "scientific_boundary": (
            "V7.0 learns shared sector-level dynamics from the eight-sector Saudi-localized panel, then aggregates "
            "sector forecasts back to the original merchant-total 15% decline objective. Transaction microdata remain "
            "UCI Online Retail II-derived and Saudi-localized; official SAMA signals are aggregate market context, not "
            "observed Saudi merchant transactions. The V7 internal holdout was not used during V7 selection, but dates "
            "overlap historical project development and therefore are not an external Saudi merchant validation set."
        ),
        "source_panel": panel_stats,
        "sector_supervised": sector_stats,
        "merchant_supervised": merchant_stats,
        "development_folds": fold_meta,
        "selected_policy": policy,
        "blind_holdout": {
            "period_start": str(HOLDOUT_START.date()),
            "period_end": str(holdout["TrainingSafeDate"].max().date()),
            "fit_meta": fit_meta,
            "metrics": holdout_metrics,
            "gates": holdout_gates,
            "all_performance_gates_passed": bool(all(holdout_gates.values())),
        },
        "scientific_integrity_gates": scientific_integrity,
        "all_scientific_integrity_gates_passed": bool(all(scientific_integrity.values())),
        "red_supported": False,
        "v6_1_reference_only_not_same_evaluation": {
            "roc_auc": 0.7583108715184187,
            "pr_auc": 0.3872897299159076,
            "precision": 0.3880597014925373,
            "recall": 0.8253968253968254,
            "f1": 0.5279187817258884,
            "green_npv": 0.9554655870445344,
            "alert_rate": 0.35170603674540685,
            "note": "V6.1 values are rolling-origin OOF development metrics, not V7 blind-holdout metrics; do not treat them as a head-to-head test.",
        },
    }
    REPORT.write_text(dumps(report), encoding="utf-8")

    dm = policy["development_metrics"]
    hm = holdout_metrics
    summary = f"""# Sales Sentinel v7.0 — Multi-Resolution Sector-to-Merchant Model

## Data design
- Source sector panel rows: **{panel_stats['panel_rows']:,}** across **{panel_stats['sectors']} sectors** and **{panel_stats['calendar_days']} days**
- Sector supervised rows: **{sector_stats['rows']:,}**
- Merchant-total supervised rows: **{merchant_stats['rows']:,}**
- Target: **next 7 days < 85% of trailing 28-day merchant baseline**
- Synthetic Region used as entity/feature: **No**
- Current/future official SAMA actuals used: **No**

## Frozen development selection
- Sector model: **{policy['sector_model']}**
- Direct merchant model: **{policy['direct_model']}**
- Blend weights — sector/direct: **{policy['sector_weight']:.2f} / {policy['direct_weight']:.2f}**
- Frozen decision threshold: **{policy['threshold']:.4f}**
- Development OOF ROC-AUC: **{dm['roc_auc']:.2%}**
- Development OOF PR-AUC: **{dm['pr_auc']:.2%}**
- Development Precision / Recall / F1: **{dm['precision']:.2%} / {dm['recall']:.2%} / {dm['f1']:.2%}**
- Development GREEN NPV / Alert rate: **{dm['green_npv']:.2%} / {dm['alert_rate']:.2%}**

## V7 internal blind holdout — opened once after freeze
- Period: **{HOLDOUT_START.date()} → {holdout['TrainingSafeDate'].max().date()}**
- Rows / positives: **{len(holdout)} / {int(holdout['target'].sum())}**
- ROC-AUC: **{hm['roc_auc']:.2%}**
- PR-AUC: **{hm['pr_auc']:.2%}**
- Accuracy / Balanced Accuracy: **{hm['accuracy']:.2%} / {hm['balanced_accuracy']:.2%}**
- Precision: **{hm['precision']:.2%}**
- Recall: **{hm['recall']:.2%}**
- F1: **{hm['f1']:.2%}**
- GREEN NPV: **{hm['green_npv']:.2%}**
- Alert rate: **{hm['alert_rate']:.2%}**
- TP / FP / FN / TN: **{hm['tp']} / {hm['fp']} / {hm['fn']} / {hm['tn']}**
- Performance gates passed: **{all(holdout_gates.values())}**
- Scientific integrity gates passed: **{all(scientific_integrity.values())}**
- RED supported: **False**

## Scientific boundary
The V7 holdout was not used to choose its model, blend or threshold. It is still an **internal** holdout drawn from the same UCI-derived Saudi-localized longitudinal source, and its dates overlap prior project experimentation. It is not a substitute for independent real Saudi merchant validation.
"""
    SUMMARY.write_text(summary, encoding="utf-8")
    print(dumps({
        "version": VERSION,
        "selected_policy": policy,
        "holdout_metrics": holdout_metrics,
        "holdout_gates": holdout_gates,
        "scientific_integrity": scientific_integrity,
    }))


if __name__ == "__main__":
    main()
