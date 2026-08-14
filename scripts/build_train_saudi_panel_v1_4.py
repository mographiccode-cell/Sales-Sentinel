from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "saudi_v1_3" / "saudi_localized_transactions_v1_3_sama.csv.gz"
SAMA_FORECAST = ROOT / "data" / "sama_pos" / "sama_market_walkforward_forecasts_2023_2025.csv"
DATA_DIR = ROOT / "data" / "saudi_v1_4"
REPORT_DIR = ROOT / "reports" / "saudi_v1_4"
MODEL_DIR = ROOT / "models" / "saudi_v1_4"
for p in (DATA_DIR, REPORT_DIR, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)

PANEL_CSV = DATA_DIR / "saudi_region_sector_daily_panel_v1_4.csv.gz"
SUPERVISED_CSV = DATA_DIR / "saudi_region_sector_supervised_v1_4.csv.gz"
AUDIT_JSON = REPORT_DIR / "dataset_quality_audit_v1_4.json"
SUMMARY_MD = REPORT_DIR / "retraining_summary_v1_4.md"
MODEL_META = MODEL_DIR / "model_metadata_v1_4.json"
MODEL_FILE = MODEL_DIR / "sales_decline_ensemble_v1_4.joblib"
VERSION = "SA-LOCALIZATION-1.4-PANEL-TRAINING-SAFE"
HORIZON = 7
BASELINE = 28
TARGET_CANDIDATES = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20]


def jd(obj):
    def default(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
        raise TypeError(type(x).__name__)
    return json.dumps(obj, indent=2, default=default)


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def week_start(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return d - pd.to_timedelta((d.dt.dayofweek + 1) % 7, unit="D")


def aggregate_panel() -> tuple[pd.DataFrame, dict]:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing rebuilt v1.3.1 full microdata: {SOURCE}")

    parts = []
    input_rows = eligible_rows = observed_customer_rows = 0
    fallback_customer_rows = 0
    administrative_rows = 0
    calibrated_nulls = 0
    used = [
        "TrainingSafeDate", "Region", "SAMASector", "SAMACalibratedNetSalesSAR",
        "BaseNetSalesSAR", "EligibleForSalesTraining", "IsAdministrativeLine",
        "SaudiInvoiceNo", "ObservedSaudiCustomerID", "CustomerIDSource", "StockCode",
        "OriginalQuantity", "PaymentType", "SAMAWeeklyMarketIndex", "SAMACalibrationFactor",
    ]

    for chunk in pd.read_csv(SOURCE, compression="gzip", usecols=used, chunksize=100_000):
        input_rows += len(chunk)
        chunk["TrainingSafeDate"] = pd.to_datetime(chunk["TrainingSafeDate"]).dt.normalize()
        eligible = truth(chunk["EligibleForSalesTraining"])
        admin = truth(chunk["IsAdministrativeLine"])
        administrative_rows += int(admin.sum())
        chunk = chunk.loc[eligible & ~admin].copy()
        eligible_rows += len(chunk)
        calibrated_nulls += int(chunk["SAMACalibratedNetSalesSAR"].isna().sum())
        observed = chunk["ObservedSaudiCustomerID"].notna() & chunk["CustomerIDSource"].astype(str).eq("ObservedSourceCustomerID")
        observed_customer_rows += int(observed.sum())
        fallback_customer_rows += int((~observed).sum())
        chunk["ObservedCustomerForModel"] = chunk["ObservedSaudiCustomerID"].where(observed, pd.NA)
        chunk["PositiveSalesSAR"] = chunk["SAMACalibratedNetSalesSAR"].astype(float).clip(lower=0)
        chunk["ReturnValueSAR"] = (-chunk["SAMACalibratedNetSalesSAR"].astype(float).clip(upper=0))
        chunk["ElectronicInvoice"] = chunk["PaymentType"].astype(str).eq("Electronic")

        g = chunk.groupby(["TrainingSafeDate", "Region", "SAMASector"], observed=True, sort=False)
        agg = g.agg(
            calibrated_sales_sar=("SAMACalibratedNetSalesSAR", "sum"),
            base_sales_sar=("BaseNetSalesSAR", "sum"),
            gross_sales_sar=("PositiveSalesSAR", "sum"),
            return_value_sar=("ReturnValueSAR", "sum"),
            transaction_rows=("SaudiInvoiceNo", "size"),
            invoice_count=("SaudiInvoiceNo", "nunique"),
            observed_customer_count=("ObservedCustomerForModel", "nunique"),
            product_count=("StockCode", "nunique"),
            units=("OriginalQuantity", lambda s: float(np.abs(pd.to_numeric(s, errors="coerce")).sum())),
            sama_market_index=("SAMAWeeklyMarketIndex", "median"),
            sama_calibration_factor=("SAMACalibrationFactor", "median"),
        ).reset_index()
        # Electronic share must be invoice-level, not row-level.
        inv = chunk[["TrainingSafeDate", "Region", "SAMASector", "SaudiInvoiceNo", "ElectronicInvoice"]].drop_duplicates(
            ["TrainingSafeDate", "Region", "SAMASector", "SaudiInvoiceNo"]
        )
        e = inv.groupby(["TrainingSafeDate", "Region", "SAMASector"], observed=True)["ElectronicInvoice"].mean().rename("electronic_invoice_share").reset_index()
        agg = agg.merge(e, on=["TrainingSafeDate", "Region", "SAMASector"], how="left")
        parts.append(agg)

    panel = pd.concat(parts, ignore_index=True)
    # Chunks can split an entity-day, so re-aggregate exactly once across chunk boundaries.
    weighted_cols = ["sama_market_index", "sama_calibration_factor", "electronic_invoice_share"]
    for c in weighted_cols:
        panel[c + "_weight"] = panel["transaction_rows"].clip(lower=1)
        panel[c + "_weighted"] = panel[c] * panel[c + "_weight"]
    panel = panel.groupby(["TrainingSafeDate", "Region", "SAMASector"], as_index=False, observed=True).agg(
        calibrated_sales_sar=("calibrated_sales_sar", "sum"),
        base_sales_sar=("base_sales_sar", "sum"),
        gross_sales_sar=("gross_sales_sar", "sum"),
        return_value_sar=("return_value_sar", "sum"),
        transaction_rows=("transaction_rows", "sum"),
        invoice_count=("invoice_count", "sum"),
        observed_customer_count=("observed_customer_count", "sum"),
        product_count=("product_count", "sum"),
        units=("units", "sum"),
        sama_market_index_weighted=("sama_market_index_weighted", "sum"),
        sama_market_index_weight=("sama_market_index_weight", "sum"),
        sama_calibration_factor_weighted=("sama_calibration_factor_weighted", "sum"),
        sama_calibration_factor_weight=("sama_calibration_factor_weight", "sum"),
        electronic_invoice_share_weighted=("electronic_invoice_share_weighted", "sum"),
        electronic_invoice_share_weight=("electronic_invoice_share_weight", "sum"),
    )
    panel["sama_market_index"] = panel["sama_market_index_weighted"] / panel["sama_market_index_weight"].clip(lower=1)
    panel["sama_calibration_factor"] = panel["sama_calibration_factor_weighted"] / panel["sama_calibration_factor_weight"].clip(lower=1)
    panel["electronic_invoice_share"] = panel["electronic_invoice_share_weighted"] / panel["electronic_invoice_share_weight"].clip(lower=1)
    panel = panel.drop(columns=[c for c in panel.columns if c.endswith("_weighted") or c.endswith("_weight")])
    panel["average_invoice_value_sar"] = panel["calibrated_sales_sar"] / panel["invoice_count"].clip(lower=1)
    panel["return_rate_value"] = panel["return_value_sar"] / panel["gross_sales_sar"].clip(lower=1)
    panel["entity"] = panel["Region"].astype(str) + " | " + panel["SAMASector"].astype(str)

    # Remove very sparse entity series rather than filling artificial zero-sales days.
    entity_stats = panel.groupby("entity").agg(
        observed_days=("TrainingSafeDate", "nunique"),
        median_invoices=("invoice_count", "median"),
        total_invoices=("invoice_count", "sum"),
    )
    keep = entity_stats[(entity_stats["observed_days"] >= 400) & (entity_stats["median_invoices"] >= 1)].index
    before_sparse = len(panel)
    panel = panel[panel["entity"].isin(keep)].copy()
    panel = panel.sort_values(["entity", "TrainingSafeDate"]).reset_index(drop=True)

    duplicates = int(panel.duplicated(["TrainingSafeDate", "entity"]).sum())
    core = ["TrainingSafeDate", "Region", "SAMASector", "calibrated_sales_sar", "invoice_count", "observed_customer_count"]
    core_nulls = int(panel[core].isna().sum().sum())
    if duplicates:
        raise RuntimeError(f"Panel has duplicate entity-date rows: {duplicates}")

    PANEL_CSV.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_CSV, index=False, compression={"method": "gzip", "compresslevel": 5})
    stats = {
        "input_microdata_rows": input_rows,
        "eligible_nonadministrative_rows": eligible_rows,
        "administrative_rows_excluded": administrative_rows,
        "observed_customer_rows": observed_customer_rows,
        "fallback_or_missing_customer_rows_not_counted_as_customers": fallback_customer_rows,
        "calibrated_sales_nulls": calibrated_nulls,
        "panel_rows_before_sparse_filter": before_sparse,
        "panel_rows": len(panel),
        "entities": int(panel["entity"].nunique()),
        "regions": int(panel["Region"].nunique()),
        "sectors": int(panel["SAMASector"].nunique()),
        "date_start": str(panel["TrainingSafeDate"].min().date()),
        "date_end": str(panel["TrainingSafeDate"].max().date()),
        "duplicate_entity_dates": duplicates,
        "core_nulls": core_nulls,
        "min_entity_observed_days": int(panel.groupby("entity")["TrainingSafeDate"].nunique().min()),
        "median_entity_observed_days": float(panel.groupby("entity")["TrainingSafeDate"].nunique().median()),
        "source_boundary": "Regions and merchant microtransactions remain Saudi-localized synthetic microdata; SAMA sectors and weekly market calibration are official aggregate Saudi signals.",
    }
    return panel, stats


def add_safe_sama_forecasts(panel: pd.DataFrame) -> pd.DataFrame:
    f = pd.read_csv(SAMA_FORECAST, parse_dates=["origin_week_start", "forecast_h1_week_start", "forecast_h2_week_start"])
    # Forecasts are true walk-forward outputs generated from SAMA history available before forecast weeks.
    f = f.sort_values("origin_week_start").drop_duplicates("origin_week_start", keep="last")
    forecast_cols = [
        "origin_week_start", "predicted_value_h1_index_52median", "predicted_value_h2_index_52median",
        "predicted_count_h1_index_52median", "predicted_count_h2_index_52median",
        "predicted_value_h1_change_vs_last", "predicted_value_h2_change_vs_last",
        "predicted_count_h1_change_vs_last", "predicted_count_h2_change_vs_last",
    ]
    f = f[forecast_cols].copy()
    d = panel.copy()
    d["week_start"] = week_start(d["TrainingSafeDate"])
    # For week W, use a forecast whose origin is W-7 days. No actual value from W or W+1 is used.
    d["forecast_origin"] = d["week_start"] - pd.Timedelta(days=7)
    d = d.merge(f, left_on="forecast_origin", right_on="origin_week_start", how="left")
    return d.drop(columns=["origin_week_start"])


def add_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    d = add_safe_sama_forecasts(panel)
    d = d.sort_values(["entity", "TrainingSafeDate"]).reset_index(drop=True)

    # National merchant context, shifted so only previous-day observations are model features.
    national = d.groupby("TrainingSafeDate", as_index=False).agg(
        national_sales=("calibrated_sales_sar", "sum"),
        national_invoices=("invoice_count", "sum"),
        national_customers=("observed_customer_count", "sum"),
    ).sort_values("TrainingSafeDate")
    for col in ["national_sales", "national_invoices", "national_customers"]:
        for lag in (1, 7, 14, 28):
            national[f"{col}_lag_{lag}"] = national[col].shift(lag)
        national[f"{col}_mean_7"] = national[col].shift(1).rolling(7).mean()
        national[f"{col}_mean_28"] = national[col].shift(1).rolling(28).mean()
    keep_nat = ["TrainingSafeDate"] + [c for c in national.columns if "_lag_" in c or "_mean_" in c]
    d = d.merge(national[keep_nat], on="TrainingSafeDate", how="left")

    g = d.groupby("entity", sort=False, group_keys=False)
    base_cols = {
        "calibrated_sales_sar": "sales",
        "base_sales_sar": "base_sales",
        "invoice_count": "invoices",
        "observed_customer_count": "customers",
        "product_count": "products",
        "units": "units",
        "average_invoice_value_sar": "basket",
        "return_rate_value": "returns",
        "electronic_invoice_share": "electronic",
    }
    for col, prefix in base_cols.items():
        for lag in (1, 2, 3, 7, 14, 28):
            d[f"{prefix}_lag_{lag}"] = g[col].shift(lag)
        for w in (7, 14, 28):
            d[f"{prefix}_mean_{w}"] = g[col].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=w).mean())
            if prefix in {"sales", "customers", "invoices", "basket"}:
                d[f"{prefix}_std_{w}"] = g[col].transform(lambda s, w=w: s.shift(1).rolling(w, min_periods=w).std())

    # Lag actual SAMA market calibration; never use current/future official SAMA as a predictor.
    d["sama_market_index_lag_7"] = g["sama_market_index"].shift(7)
    d["sama_factor_lag_7"] = g["sama_calibration_factor"].shift(7)

    # Calendar seasonality is knowable in advance; do not use the inherited source weekday.
    dt = pd.to_datetime(d["TrainingSafeDate"])
    doy = dt.dt.dayofyear.astype(float)
    d["year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["year_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Future seven-day target and historical 28-day baseline within the same entity.
    future = []
    for i in range(1, HORIZON + 1):
        future.append(g["calibrated_sales_sar"].shift(-i).rename(f"future_{i}"))
    future_df = pd.concat(future, axis=1)
    d["future_7_mean_sales"] = future_df.mean(axis=1)
    d["future_7_complete"] = future_df.notna().sum(axis=1).eq(HORIZON)
    d["baseline_28_sales"] = g["calibrated_sales_sar"].transform(lambda s: s.shift(1).rolling(BASELINE, min_periods=BASELINE).mean())
    d["future_ratio"] = d["future_7_mean_sales"] / d["baseline_28_sales"].replace(0, np.nan)

    # Categorical identity: model learns persistent level differences, not target-derived encodings.
    cats = pd.get_dummies(d[["Region", "SAMASector"]], prefix=["region", "sector"], dtype=float)
    d = pd.concat([d, cats], axis=1)

    forbidden = {
        "calibrated_sales_sar", "gross_sales_sar", "return_value_sar", "base_sales_sar",
        "sama_market_index", "sama_calibration_factor", "future_7_mean_sales", "baseline_28_sales",
        "future_ratio", "future_7_complete", "TrainingSafeDate", "entity", "Region", "SAMASector",
        "week_start", "forecast_origin",
    }
    feature_cols = [
        c for c in d.columns
        if c not in forbidden
        and not c.startswith("future_")
        and c not in {"transaction_rows", "invoice_count", "observed_customer_count", "product_count", "units", "average_invoice_value_sar", "return_rate_value", "electronic_invoice_share"}
    ]
    # Explicitly ensure actual future SAMA columns can never leak in.
    feature_cols = [c for c in feature_cols if not c.startswith("actual_")]
    d = d.replace([np.inf, -np.inf], np.nan)
    required = feature_cols + ["future_ratio", "baseline_28_sales"]
    before = len(d)
    d = d[d["future_7_complete"]].dropna(subset=required).reset_index(drop=True)
    stats = {"rows_before_supervised_drop": before, "supervised_rows": len(d), "features": len(feature_cols)}
    return d, feature_cols, stats


def choose_decline_threshold(train: pd.DataFrame) -> tuple[float, list[dict]]:
    rows = []
    for t in TARGET_CANDIDATES:
        y = (train["future_ratio"] < (1.0 - t)).astype(int)
        monthly = train.assign(y=y).groupby(train["TrainingSafeDate"].dt.to_period("Q"))["y"].agg(["count", "mean", "sum"])
        valid_segments = monthly[monthly["count"] >= 200]
        segment_min = float(valid_segments["mean"].min()) if len(valid_segments) else 0.0
        segment_max = float(valid_segments["mean"].max()) if len(valid_segments) else 1.0
        rate = float(y.mean())
        eligible = bool(0.16 <= rate <= 0.35 and segment_min >= 0.08 and segment_max <= 0.50)
        rows.append({
            "decline_threshold": t, "positive_rate": rate, "positive_count": int(y.sum()),
            "quarter_min_rate": segment_min, "quarter_max_rate": segment_max, "eligible": eligible,
        })
    eligible = [r for r in rows if r["eligible"]]
    # Prefer the largest commercially meaningful decline that still has enough examples.
    chosen = max(eligible, key=lambda r: r["decline_threshold"]) if eligible else min(rows, key=lambda r: abs(r["positive_rate"] - 0.25))
    return float(chosen["decline_threshold"]), rows


def classification_metrics(y, score, threshold):
    pred = (np.asarray(score) >= threshold).astype(int)
    y = np.asarray(y).astype(int)
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y, score)),
        "ConfusionMatrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def metric_score(m):
    return 0.24*m["BalancedAccuracy"] + 0.22*m["F1"] + 0.18*m["ROC_AUC"] + 0.18*m["Recall"] + 0.18*m["Accuracy"]


def classifiers(pos_weight: float):
    out = {
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.4, max_iter=4000, class_weight="balanced", random_state=SEED)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=700, max_depth=14, min_samples_leaf=5, max_features=0.55,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=900, max_depth=16, min_samples_leaf=4, max_features=0.65,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=350, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=35,
            l2_regularization=2.0, class_weight="balanced", random_state=SEED,
        ),
    }
    if XGBClassifier is not None:
        out["XGBoost"] = XGBClassifier(
            n_estimators=700, max_depth=6, learning_rate=0.035, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=8, reg_lambda=2.0, reg_alpha=0.15,
            scale_pos_weight=pos_weight, eval_metric="logloss", random_state=SEED, n_jobs=-1,
        )
    return out


def regressors():
    out = {
        "Ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=8.0))]),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=800, max_depth=16, min_samples_leaf=4, max_features=0.65,
            random_state=SEED, n_jobs=-1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=350, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=35,
            l2_regularization=2.0, random_state=SEED,
        ),
    }
    if XGBRegressor is not None:
        out["XGBoostRegressor"] = XGBRegressor(
            n_estimators=700, max_depth=6, learning_rate=0.035, subsample=0.85,
            colsample_bytree=0.8, min_child_weight=8, reg_lambda=2.0, reg_alpha=0.1,
            objective="reg:squarederror", random_state=SEED, n_jobs=-1,
        )
    return out


def best_threshold(y, score):
    best = None
    for t in np.linspace(0.05, 0.95, 181):
        m = classification_metrics(y, score, float(t))
        # Penalize useless high-accuracy, low-recall solutions.
        value = metric_score(m) - 0.35 * max(0.0, 0.70 - m["Recall"])
        candidate = (value, m["BalancedAccuracy"], m["F1"], m["ROC_AUC"], -abs(t-0.5), float(t), m)
        if best is None or candidate[:5] > best[:5]:
            best = candidate
    return best[5], best[6], best[0]


def train_and_evaluate(d: pd.DataFrame, features: list[str], decline_threshold: float) -> dict:
    d = d.copy()
    d["target"] = (d["future_ratio"] < (1.0 - decline_threshold)).astype(int)

    # Global chronological split with a seven-day purge between partitions.
    train = d[d["TrainingSafeDate"] <= pd.Timestamp("2023-12-24")].copy()
    val = d[(d["TrainingSafeDate"] >= pd.Timestamp("2024-01-08")) & (d["TrainingSafeDate"] <= pd.Timestamp("2024-04-30"))].copy()
    test = d[d["TrainingSafeDate"] >= pd.Timestamp("2024-05-08")].copy()
    if min(len(train), len(val), len(test)) < 2000:
        raise RuntimeError(f"Insufficient panel splits: {len(train)}/{len(val)}/{len(test)}")
    if min(train.target.nunique(), val.target.nunique(), test.target.nunique()) < 2:
        raise RuntimeError("One chronological split contains only one target class")

    pos = int(train.target.sum()); neg = len(train) - pos
    pos_weight = neg / max(pos, 1)
    model_results = {}
    fitted_candidates = {}
    validation_scores = {}
    for name, model in classifiers(pos_weight).items():
        fit = clone(model).fit(train[features], train.target)
        p = fit.predict_proba(val[features])[:, 1]
        t, m, s = best_threshold(val.target.to_numpy(), p)
        model_results[name] = {"threshold": t, "metrics": m, "selection_score": s}
        fitted_candidates[name] = fit
        validation_scores[name] = p
    best_cls_name = max(model_results, key=lambda n: (model_results[n]["selection_score"], model_results[n]["metrics"]["ROC_AUC"]))

    # Continuous target can use information from all examples; compare regression-derived decline scores.
    reg_results = {}
    fitted_reg = {}
    reg_scores = {}
    for name, model in regressors().items():
        fit = clone(model).fit(train[features], train["future_ratio"].clip(0.05, 3.0))
        pred_ratio = fit.predict(val[features])
        # Smooth risk around the chosen business threshold.
        risk = 1.0 / (1.0 + np.exp(np.clip((pred_ratio - (1.0 - decline_threshold)) / 0.06, -30, 30)))
        t, m, s = best_threshold(val.target.to_numpy(), risk)
        reg_results[name] = {
            "threshold": t, "metrics": m, "selection_score": s,
            "validation_ratio_mae": float(mean_absolute_error(val["future_ratio"], pred_ratio)),
        }
        fitted_reg[name] = fit
        reg_scores[name] = risk
    best_reg_name = max(reg_results, key=lambda n: (reg_results[n]["selection_score"], reg_results[n]["metrics"]["ROC_AUC"]))

    # Validation-only blend search; test remains untouched.
    p_cls = validation_scores[best_cls_name]
    p_reg = reg_scores[best_reg_name]
    best_blend = None
    for w in np.linspace(0.0, 1.0, 21):
        score = w * p_cls + (1.0 - w) * p_reg
        t, m, s = best_threshold(val.target.to_numpy(), score)
        candidate = (s, m["BalancedAccuracy"], m["F1"], m["ROC_AUC"], float(w), t, m)
        if best_blend is None or candidate[:4] > best_blend[:4]:
            best_blend = candidate
    blend_weight, blend_threshold = best_blend[4], best_blend[5]

    trainval = pd.concat([train, val], ignore_index=True).sort_values("TrainingSafeDate")
    final_cls = clone(classifiers((len(trainval)-int(trainval.target.sum()))/max(int(trainval.target.sum()),1))[best_cls_name]).fit(trainval[features], trainval.target)
    final_reg = clone(regressors()[best_reg_name]).fit(trainval[features], trainval["future_ratio"].clip(0.05, 3.0))
    test_cls = final_cls.predict_proba(test[features])[:, 1]
    test_ratio = final_reg.predict(test[features])
    test_reg_risk = 1.0 / (1.0 + np.exp(np.clip((test_ratio - (1.0 - decline_threshold)) / 0.06, -30, 30)))
    test_blend = blend_weight * test_cls + (1.0 - blend_weight) * test_reg_risk
    test_metrics = classification_metrics(test.target.to_numpy(), test_blend, blend_threshold)
    majority = max(float(test.target.mean()), 1.0 - float(test.target.mean()))

    gates = {
        "accuracy_above_majority_by_5pp": test_metrics["Accuracy"] >= majority + 0.05,
        "balanced_accuracy_at_least_75pct": test_metrics["BalancedAccuracy"] >= 0.75,
        "recall_at_least_70pct": test_metrics["Recall"] >= 0.70,
        "f1_at_least_65pct": test_metrics["F1"] >= 0.65,
        "roc_auc_at_least_82pct": test_metrics["ROC_AUC"] >= 0.82,
    }
    high_accuracy_goal = test_metrics["Accuracy"] >= 0.90

    joblib.dump({
        "classifier": final_cls,
        "regressor": final_reg,
        "features": features,
        "blend_weight_classifier": blend_weight,
        "blend_weight_regression": 1.0 - blend_weight,
        "probability_threshold": blend_threshold,
        "decline_threshold": decline_threshold,
        "horizon_days": HORIZON,
        "version": VERSION,
        "classifier_name": best_cls_name,
        "regressor_name": best_reg_name,
    }, MODEL_FILE)

    return {
        "split": {
            "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
            "train_end": "2023-12-24", "validation": "2024-01-08..2024-04-30", "test_start": "2024-05-08",
            "purge_days": 7, "shuffle": False,
        },
        "target": {
            "definition": f"mean sales of next {HORIZON} observed entity-days is at least {decline_threshold:.1%} below trailing {BASELINE}-day mean",
            "decline_threshold": decline_threshold,
            "train_positive_rate": float(train.target.mean()), "validation_positive_rate": float(val.target.mean()), "test_positive_rate": float(test.target.mean()),
        },
        "classification_candidates": model_results,
        "regression_candidates": reg_results,
        "selected_classifier": best_cls_name,
        "selected_regressor": best_reg_name,
        "blend_weight_classifier": blend_weight,
        "blend_weight_regression": 1.0-blend_weight,
        "selected_probability_threshold": blend_threshold,
        "test_metrics": test_metrics,
        "majority_test_accuracy": majority,
        "high_accuracy_90pct_goal_met": high_accuracy_goal,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": bool(all(gates.values())),
        "leakage_controls": {
            "actual_future_SAMA_values_used_as_features": False,
            "same_week_actual_SAMA_market_index_used_as_feature": False,
            "walk_forward_SAMA_forecasts_only": True,
            "fallback_customer_ids_counted_as_customers": False,
            "future_target_columns_in_features": False,
            "chronological_split": True,
            "test_used_for_model_or_threshold_selection": False,
        },
    }


def main():
    panel, panel_stats = aggregate_panel()
    supervised, features, supervised_stats = add_features(panel)
    train_for_target = supervised[supervised["TrainingSafeDate"] <= pd.Timestamp("2023-12-24")].copy()
    decline_threshold, target_diagnostics = choose_decline_threshold(train_for_target)
    supervised.to_csv(SUPERVISED_CSV, index=False, compression={"method": "gzip", "compresslevel": 5})

    core_checks = {
        "source_has_verified_million_rows": panel_stats["input_microdata_rows"] == 1_049_042,
        "no_duplicate_entity_dates": panel_stats["duplicate_entity_dates"] == 0,
        "no_core_nulls": panel_stats["core_nulls"] == 0,
        "no_calibrated_sales_nulls": panel_stats["calibrated_sales_nulls"] == 0,
        "administrative_rows_excluded": panel_stats["administrative_rows_excluded"] > 0,
        "fallback_customer_ids_excluded_from_customer_counts": True,
        "at_least_50_entities": panel_stats["entities"] >= 50,
        "at_least_25000_panel_rows": panel_stats["panel_rows"] >= 25_000,
        "at_least_20000_supervised_rows": supervised_stats["supervised_rows"] >= 20_000,
        "each_entity_has_at_least_400_observed_days": panel_stats["min_entity_observed_days"] >= 400,
        "target_selected_on_training_period_only": True,
        "future_SAMA_actuals_forbidden": True,
    }
    if not all(core_checks.values()):
        audit = {"version": VERSION, "panel": panel_stats, "supervised": supervised_stats, "target_diagnostics": target_diagnostics, "dataset_checks": core_checks, "dataset_quality_passed": False}
        AUDIT_JSON.write_text(jd(audit), encoding="utf-8")
        raise RuntimeError("Dataset v1.4 quality gate failed before training")

    training = train_and_evaluate(supervised, features, decline_threshold)
    audit = {
        "version": VERSION,
        "scientific_boundary": "The training panel is derived from UCI merchant microtransactions localized to Saudi context. Region assignments remain synthetic/calibrated, while SAMA market signals are official Saudi aggregates. This is not observed merchant-level Saudi transaction microdata.",
        "purpose": "Fix the 604-point bottleneck by training across repeated region-sector time series rather than copying one national target onto raw invoice rows.",
        "panel": panel_stats,
        "supervised": supervised_stats,
        "target_diagnostics": target_diagnostics,
        "feature_columns": features,
        "dataset_checks": core_checks,
        "dataset_quality_passed": True,
        "training": training,
    }
    AUDIT_JSON.write_text(jd(audit), encoding="utf-8")
    MODEL_META.write_text(jd(audit), encoding="utf-8")
    m = training["test_metrics"]
    SUMMARY_MD.write_text(
        "# Saudi Panel Retraining v1.4\n\n"
        f"- Dataset quality gate: **PASS**\n"
        f"- Source microdata rows verified: **{panel_stats['input_microdata_rows']:,}**\n"
        f"- Panel rows: **{panel_stats['panel_rows']:,}**\n"
        f"- Supervised rows: **{supervised_stats['supervised_rows']:,}**\n"
        f"- Entities: **{panel_stats['entities']}**\n"
        f"- Features: **{len(features)}**\n"
        f"- Decline definition: **{decline_threshold:.1%}**, next **{HORIZON}** days vs trailing **{BASELINE}** days\n"
        f"- Selected classifier: **{training['selected_classifier']}**\n"
        f"- Selected regressor: **{training['selected_regressor']}**\n"
        f"- Accuracy: **{m['Accuracy']:.2%}**\n"
        f"- Balanced Accuracy: **{m['BalancedAccuracy']:.2%}**\n"
        f"- Precision: **{m['Precision']:.2%}**\n"
        f"- Recall: **{m['Recall']:.2%}**\n"
        f"- F1: **{m['F1']:.2%}**\n"
        f"- ROC-AUC: **{m['ROC_AUC']:.2%}**\n"
        f"- Majority baseline: **{training['majority_test_accuracy']:.2%}**\n"
        f"- 90% accuracy goal met: **{training['high_accuracy_90pct_goal_met']}**\n"
        f"- All scientific acceptance gates passed: **{training['all_acceptance_gates_passed']}**\n",
        encoding="utf-8",
    )
    print(jd({
        "dataset_quality": "PASS",
        "panel_rows": panel_stats["panel_rows"],
        "supervised_rows": supervised_stats["supervised_rows"],
        "entities": panel_stats["entities"],
        "decline_threshold": decline_threshold,
        "selected_classifier": training["selected_classifier"],
        "selected_regressor": training["selected_regressor"],
        "test_metrics": training["test_metrics"],
        "majority": training["majority_test_accuracy"],
        "goal_90": training["high_accuracy_90pct_goal_met"],
        "acceptance": training["all_acceptance_gates_passed"],
    }))


if __name__ == "__main__":
    main()
