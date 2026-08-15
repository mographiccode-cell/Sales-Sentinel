from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import build_saudi_sama_calibrated_v1_3 as loc

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-MERCHANT-SAMA-HYBRID-4.0"
SEED = 42
DECLINE = 0.20
FULL_GZ = ROOT / "artifacts" / "saudi_v1_3" / "saudi_localized_transactions_v1_3_sama.csv.gz"
SAMA_FORECAST = ROOT / "data" / "sama_pos" / "sama_sector_walkforward_forecasts_2023_2025.csv"
OUT = ROOT / "reports" / "merchant_sama_hybrid_v4"
MOD = ROOT / "models" / "merchant_sama_hybrid_v4"
DATA = ROOT / "data" / "merchant_v4"
for p in (OUT, MOD, DATA):
    p.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
MODEL = MOD / "merchant_sama_hybrid_v4.joblib"
FEATURE_PANEL = DATA / "merchant_sector_daily_features_v4.csv"

TRAIN_END = pd.Timestamp("2023-10-24")
VAL_START = pd.Timestamp("2023-11-01")
VAL_END = pd.Timestamp("2024-01-24")
TEST_START = pd.Timestamp("2024-02-01")

RAMADAN = [("2023-03-23", "2023-04-20"), ("2024-03-11", "2024-04-09")]
EID_FITR = [("2023-04-21", "2023-04-23"), ("2024-04-10", "2024-04-12")]
HAJJ = [("2023-06-19", "2023-06-30"), ("2024-06-07", "2024-06-19")]
EID_ADHA = [("2023-06-28", "2023-07-01"), ("2024-06-16", "2024-06-19")]


def in_ranges(dates: pd.Series, ranges: list[tuple[str, str]]) -> np.ndarray:
    result = np.zeros(len(dates), dtype=bool)
    for start, end in ranges:
        result |= dates.between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()
    return result


def week_start(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series)
    return dates - pd.to_timedelta((dates.dt.dayofweek + 1) % 7, unit="D")


def aggregate_microdata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Full v1.3 microdata missing: {path}. Rebuild v1.3.1 first.")

    numeric = defaultdict(lambda: defaultdict(float))
    invoices = defaultdict(set)
    electronic_invoices = defaultdict(set)
    customers = defaultdict(set)
    products = defaultdict(set)
    source_rows = 0
    eligible_rows = 0

    usecols = [
        "TrainingSafeDate", "ProductCategoryCOICOP", "SAMASector", "SAMACalibratedNetSalesSAR",
        "OriginalQuantity", "SaudiInvoiceNo", "PaymentType", "StockCode", "ObservedSaudiCustomerID",
        "EligibleForSalesTraining",
    ]
    for chunk in pd.read_csv(path, compression="gzip", usecols=usecols, chunksize=120_000):
        source_rows += len(chunk)
        eligible = chunk["EligibleForSalesTraining"].astype(str).str.lower().isin(["true", "1"])
        chunk = chunk.loc[eligible].copy()
        eligible_rows += len(chunk)
        chunk["TrainingSafeDate"] = pd.to_datetime(chunk["TrainingSafeDate"]).dt.normalize()
        chunk["ProductCategoryCOICOP"] = chunk["ProductCategoryCOICOP"].astype(str)
        chunk["SAMASector"] = chunk["SAMASector"].astype(str)
        chunk["net"] = pd.to_numeric(chunk["SAMACalibratedNetSalesSAR"], errors="coerce").fillna(0.0)
        chunk["qty"] = pd.to_numeric(chunk["OriginalQuantity"], errors="coerce").fillna(0.0).abs()

        for (day, category, sector), group in chunk.groupby(["TrainingSafeDate", "ProductCategoryCOICOP", "SAMASector"], sort=False):
            key = (pd.Timestamp(day), str(category), str(sector))
            vals = numeric[key]
            net = group["net"].astype(float)
            vals["net_sales_sar"] += float(net.sum())
            vals["gross_sales_sar"] += float(net.clip(lower=0).sum())
            vals["return_value_sar"] += float((-net.clip(upper=0)).sum())
            vals["units"] += float(group["qty"].sum())
            vals["line_rows"] += int(len(group))
            invoices[key].update(group["SaudiInvoiceNo"].dropna().astype(str))
            electronic_invoices[key].update(group.loc[group["PaymentType"].eq("Electronic"), "SaudiInvoiceNo"].dropna().astype(str))
            customers[key].update(group["ObservedSaudiCustomerID"].dropna().astype(str))
            products[key].update(group["StockCode"].dropna().astype(str))

    rows = []
    for key in sorted(numeric):
        day, category, sector = key
        v = numeric[key]
        inv = len(invoices[key])
        gross = float(v["gross_sales_sar"])
        net = float(v["net_sales_sar"])
        units = float(v["units"])
        rows.append({
            "date": day,
            "category": category,
            "sama_sector": sector,
            "net_sales_sar": net,
            "gross_sales_sar": gross,
            "return_value_sar": float(v["return_value_sar"]),
            "units": units,
            "line_rows": int(v["line_rows"]),
            "invoice_count": inv,
            "electronic_invoice_count": len(electronic_invoices[key]),
            "observed_customer_count": len(customers[key]),
            "unique_products": len(products[key]),
            "avg_invoice_value_sar": net / max(inv, 1),
            "avg_unit_value_sar": gross / max(units, 1.0),
            "return_rate_value": float(v["return_value_sar"]) / max(gross, 1e-9),
            "electronic_share": len(electronic_invoices[key]) / max(inv, 1),
        })

    d = pd.DataFrame(rows)
    if source_rows != 1_049_042:
        raise RuntimeError(f"Expected 1,049,042 localized source rows, found {source_rows:,}")
    if eligible_rows <= 1_000_000:
        raise RuntimeError(f"Unexpectedly few eligible rows: {eligible_rows:,}")
    d.attrs["source_rows"] = source_rows
    d.attrs["eligible_rows"] = eligible_rows
    return d


def complete_panel(raw: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(raw.date.min(), raw.date.max(), freq="D")
    mapping = raw[["category", "sama_sector"]].drop_duplicates()
    if mapping.groupby("category").sama_sector.nunique().max() != 1:
        raise RuntimeError("Category-to-SAMA-sector mapping is not stable")
    categories = mapping.category.tolist()
    grid = pd.MultiIndex.from_product([dates, categories], names=["date", "category"]).to_frame(index=False)
    d = grid.merge(raw, on=["date", "category"], how="left")
    d = d.merge(mapping, on="category", how="left", suffixes=("", "_map"))
    d["sama_sector"] = d["sama_sector"].fillna(d["sama_sector_map"])
    d = d.drop(columns=["sama_sector_map"])
    numeric = [
        "net_sales_sar", "gross_sales_sar", "return_value_sar", "units", "line_rows",
        "invoice_count", "electronic_invoice_count", "observed_customer_count", "unique_products",
        "avg_invoice_value_sar", "avg_unit_value_sar", "return_rate_value", "electronic_share",
    ]
    d[numeric] = d[numeric].fillna(0.0)
    return d.sort_values(["category", "date"]).reset_index(drop=True)


def add_sama_forecasts(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.read_csv(SAMA_FORECAST, parse_dates=["origin_week_start"])
    forbidden = [c for c in f.columns if c.startswith("actual_")]
    keep = [
        "origin_week_start", "sector",
        "predicted_value_h1_index_52median", "predicted_value_h2_index_52median",
        "predicted_count_h1_index_52median", "predicted_count_h2_index_52median",
        "predicted_value_h1_change_vs_last", "predicted_value_h2_change_vs_last",
        "predicted_count_h1_change_vs_last", "predicted_count_h2_change_vs_last",
    ]
    f = f[keep].rename(columns={"sector": "sama_sector"})
    out = d.copy()
    out["sama_week_start"] = week_start(out["date"])
    # Only the previously completed SAMA week can inform a merchant decision on the current week.
    out["sama_forecast_origin"] = out["sama_week_start"] - pd.Timedelta(days=7)
    out = out.merge(f, left_on=["sama_forecast_origin", "sama_sector"], right_on=["origin_week_start", "sama_sector"], how="left", validate="many_to_one")
    out = out.drop(columns=["origin_week_start"])
    if forbidden and any(c in out.columns for c in forbidden):
        raise RuntimeError("Future actual SAMA leaked into merchant panel")
    return out


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(float) / b.astype(float).replace(0, np.nan)


def make_features(panel: pd.DataFrame):
    d = panel.copy().sort_values(["category", "date"]).reset_index(drop=True)
    g = d.groupby("category", sort=False)
    d["sales_target_base"] = d["net_sales_sar"].clip(lower=0)
    d["baseline28_daily"] = g.sales_target_base.transform(lambda s: s.rolling(28, min_periods=28).mean())
    future = sum(g.sales_target_base.shift(-h) for h in range(1, 8))
    d["future7_sales"] = future
    d["future_ratio"] = d["future7_sales"] / (7.0 * d["baseline28_daily"].replace(0, np.nan))
    d["target"] = (d["future_ratio"] < (1.0 - DECLINE)).astype(int)

    X = pd.DataFrame(index=d.index)
    dynamic = [
        "net_sales_sar", "gross_sales_sar", "invoice_count", "observed_customer_count",
        "unique_products", "units", "avg_invoice_value_sar", "avg_unit_value_sar", "return_rate_value",
        "electronic_share",
    ]
    for col in dynamic:
        s = d[col].astype(float)
        prefix = col.replace("_sar", "")
        for w in (7, 14, 28, 56):
            mean = g[col].transform(lambda z, w=w: z.rolling(w, min_periods=w).mean())
            X[f"{prefix}_ratio_mean_{w}"] = safe_ratio(s, mean)
        for lag in (1, 7, 14, 28):
            X[f"{prefix}_change_{lag}"] = g[col].pct_change(lag)

    merchant = d.groupby("date", as_index=False).agg(
        merchant_net_sales=("net_sales_sar", "sum"),
        merchant_gross_sales=("gross_sales_sar", "sum"),
        merchant_invoices=("invoice_count", "sum"),
        merchant_customers=("observed_customer_count", "sum"),
        merchant_units=("units", "sum"),
        merchant_returns=("return_value_sar", "sum"),
    ).sort_values("date")
    for col in ["merchant_net_sales", "merchant_invoices", "merchant_customers", "merchant_units"]:
        for w in (7, 28, 56):
            merchant[f"{col}_ratio_mean_{w}"] = safe_ratio(merchant[col], merchant[col].rolling(w, min_periods=w).mean())
        merchant[f"{col}_change_7"] = merchant[col].pct_change(7)
        merchant[f"{col}_change_28"] = merchant[col].pct_change(28)
    merchant["merchant_return_rate"] = merchant.merchant_returns / merchant.merchant_gross_sales.clip(lower=1e-9)
    mcols = [c for c in merchant.columns if c != "date"]
    dm = d[["date"]].merge(merchant, on="date", how="left", validate="many_to_one")
    X = pd.concat([X, dm[mcols]], axis=1)

    total_by_day = d.groupby("date").net_sales_sar.transform("sum")
    d["category_share"] = d.net_sales_sar / total_by_day.replace(0, np.nan)
    gs = d.groupby("category", sort=False)
    X["category_share_ratio_28"] = safe_ratio(d.category_share, gs.category_share.transform(lambda s: s.rolling(28, min_periods=28).mean()))
    X["category_share_change_7"] = gs.category_share.pct_change(7)

    forecast_cols = [c for c in d.columns if c.startswith("predicted_")]
    for c in forecast_cols:
        X[f"sama_{c}"] = pd.to_numeric(d[c], errors="coerce")

    dates = pd.to_datetime(d.date)
    X["is_ramadan"] = in_ranges(dates, RAMADAN).astype(float)
    X["is_eid_fitr"] = in_ranges(dates, EID_FITR).astype(float)
    X["is_hajj"] = in_ranges(dates, HAJJ).astype(float)
    X["is_eid_adha"] = in_ranges(dates, EID_ADHA).astype(float)
    X["is_national_day_window"] = dates.dt.dayofyear.between(pd.Timestamp("2024-09-16").dayofyear, pd.Timestamp("2024-09-30").dayofyear).astype(float)
    X["is_founding_day_window"] = ((dates.dt.month == 2) & dates.dt.day.between(15, 29)).astype(float)
    X["salary_period"] = dates.dt.day.between(24, 31).astype(float)
    dow = dates.dt.dayofweek.astype(float)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    doy = dates.dt.dayofyear.astype(float)
    X["year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["year_cos"] = np.cos(2 * np.pi * doy / 365.25)

    cats = pd.get_dummies(d[["category"]], prefix="category", dtype=float)
    X = pd.concat([X, cats], axis=1).replace([np.inf, -np.inf], np.nan)
    good = X.notna().all(axis=1) & d.future_ratio.notna() & d.baseline28_daily.gt(0)
    meta = d.loc[good, ["date", "category", "sama_sector", "future_ratio", "future7_sales", "baseline28_daily", "target"]].reset_index(drop=True)
    return meta, X.loc[good].reset_index(drop=True)


def model_factories():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.20, class_weight="balanced", max_iter=5000, random_state=SEED)),
        "extra_trees": ExtraTreesClassifier(n_estimators=700, max_depth=9, min_samples_leaf=7, max_features=0.65, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "hist_gb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=14, min_samples_leaf=24, l2_regularization=8.0, random_state=SEED),
    }


def fit_classifier(model, X, y):
    if isinstance(model, HistGradientBoostingClassifier):
        pos = max(int(y.sum()), 1)
        neg = max(len(y) - pos, 1)
        sw = np.where(np.asarray(y) == 1, neg / pos, 1.0)
        return model.fit(X, y, sample_weight=sw)
    return model.fit(X, y)


def metrics(y, score, threshold):
    pred = np.asarray(score) >= threshold
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "alert_rate": float(pred.mean()),
        "green_npv": float(((np.asarray(y) == 0) & (~pred)).sum() / max((~pred).sum(), 1)),
        "tp": int(((np.asarray(y) == 1) & pred).sum()),
        "fp": int(((np.asarray(y) == 0) & pred).sum()),
        "fn": int(((np.asarray(y) == 1) & (~pred)).sum()),
        "tn": int(((np.asarray(y) == 0) & (~pred)).sum()),
    }


def choose_threshold(y, score):
    candidates = np.unique(np.r_[np.linspace(0.05, 0.95, 181), np.quantile(score, np.linspace(0.02, 0.98, 97))])
    rows = []
    for t in candidates:
        m = metrics(y, score, float(t))
        rows.append((float(t), m))
    feasible = [(t, m) for t, m in rows if m["recall"] >= 0.75 and m["alert_rate"] <= 0.50]
    pool = feasible if feasible else rows
    pool.sort(key=lambda z: (z[1]["balanced_accuracy"], z[1]["f1"], z[1]["recall"], z[1]["precision"]), reverse=True)
    return pool[0], len(feasible)


def choose_red_threshold(y, score, watch_t):
    rows = []
    for t in np.unique(np.r_[np.linspace(max(watch_t, 0.30), 0.99, 140), np.quantile(score, np.linspace(0.50, 0.995, 80))]):
        m = metrics(y, score, float(t))
        if m["tp"] + m["fp"] >= 5 and m["precision"] >= 0.70:
            rows.append((float(t), m))
    if not rows:
        return 0.99, metrics(y, score, 0.99), 0
    rows.sort(key=lambda z: (z[1]["recall"], z[1]["precision"], -z[1]["alert_rate"]), reverse=True)
    return rows[0][0], rows[0][1], len(rows)


def main():
    raw = aggregate_microdata(FULL_GZ)
    source_rows = int(raw.attrs["source_rows"])
    eligible_rows = int(raw.attrs["eligible_rows"])
    panel = complete_panel(raw)
    panel = add_sama_forecasts(panel)
    meta, X = make_features(panel)
    FEATURE_PANEL.parent.mkdir(parents=True, exist_ok=True)
    evidence = pd.concat([meta, X], axis=1)
    evidence.to_csv(FEATURE_PANEL, index=False)

    forbidden_features = [c for c in X.columns if c.startswith("actual_") or "future" in c.lower() or "target" in c.lower()]
    if forbidden_features:
        raise RuntimeError(f"Forbidden feature columns: {forbidden_features}")

    tr = meta.date <= TRAIN_END
    va = meta.date.between(VAL_START, VAL_END)
    te = meta.date >= TEST_START
    if min(int(tr.sum()), int(va.sum()), int(te.sum())) < 400:
        raise RuntimeError(f"Insufficient chronological split: {int(tr.sum())}/{int(va.sum())}/{int(te.sum())}")
    for mask, name in [(tr, "train"), (va, "validation"), (te, "test")]:
        if meta.loc[mask, "target"].nunique() != 2:
            raise RuntimeError(f"{name} split has one target class")

    ytr = meta.loc[tr, "target"].astype(int)
    yva = meta.loc[va, "target"].astype(int)
    yte = meta.loc[te, "target"].astype(int)

    candidates = {}
    fitted = {}
    val_scores = {}
    test_scores = {}
    for name, factory in model_factories().items():
        model = fit_classifier(clone(factory), X.loc[tr], ytr)
        pv = model.predict_proba(X.loc[va])[:, 1]
        pt = model.predict_proba(X.loc[te])[:, 1]
        auc = float(roc_auc_score(yva, pv))
        candidates[name] = {"validation_roc_auc": auc}
        fitted[name] = model
        val_scores[name] = pv
        test_scores[name] = pt

    # Continuous downside model adds a dense signal without using future values as features.
    reg = HistGradientBoostingRegressor(max_iter=320, learning_rate=0.03, max_leaf_nodes=14, min_samples_leaf=24, l2_regularization=8.0, random_state=SEED)
    reg.fit(X.loc[tr], meta.loc[tr, "future_ratio"].clip(0, 2.5))
    rv = reg.predict(X.loc[va])
    rt = reg.predict(X.loc[te])
    reg_risk_v = 1.0 / (1.0 + np.exp((rv - 0.80) / 0.08))
    reg_risk_t = 1.0 / (1.0 + np.exp((rt - 0.80) / 0.08))

    names = list(model_factories())
    cls_v = np.column_stack([val_scores[n] for n in names]).mean(axis=1)
    cls_t = np.column_stack([test_scores[n] for n in names]).mean(axis=1)
    blend_rows = []
    for w in (0.0, 0.15, 0.30, 0.45):
        score = (1.0 - w) * cls_v + w * reg_risk_v
        auc = float(roc_auc_score(yva, score))
        (t, m), feasible_n = choose_threshold(yva, score)
        blend_rows.append((auc, m["balanced_accuracy"], m["f1"], -w, w, t, m, feasible_n))
    blend_rows.sort(reverse=True)
    _, _, _, _, blend_w, watch_t, val_metrics, feasible_thresholds = blend_rows[0]
    val_final_score = (1.0 - blend_w) * cls_v + blend_w * reg_risk_v
    test_final_score = (1.0 - blend_w) * cls_t + blend_w * reg_risk_t
    red_t, val_red_metrics, red_candidates = choose_red_threshold(yva, val_final_score, watch_t)
    test_metrics = metrics(yte, test_final_score, watch_t)
    test_red_metrics = metrics(yte, test_final_score, red_t)

    # Fit final artifacts using train+validation only; test remains untouched for reported metrics.
    fit_mask = meta.date <= VAL_END
    yfit = meta.loc[fit_mask, "target"].astype(int)
    final_models = {name: fit_classifier(clone(factory), X.loc[fit_mask], yfit) for name, factory in model_factories().items()}
    final_reg = clone(reg).fit(X.loc[fit_mask], meta.loc[fit_mask, "future_ratio"].clip(0, 2.5))

    artifact = {
        "version": VERSION,
        "feature_columns": list(X.columns),
        "models": final_models,
        "regressor": final_reg,
        "blend_weight_regression": float(blend_w),
        "watch_threshold": float(watch_t),
        "red_threshold": float(red_t),
        "target_definition": "next 7 calendar days category net sales <80% of trailing 28-day daily mean x7",
        "source_scope": "Saudi-localized synthetic merchant microdata calibrated to official SAMA aggregates",
        "sama_external_signal": "leakage-safe v1.7 walk-forward predicted sector value/count only; no future actual SAMA",
        "training_cutoff": str(VAL_END.date()),
    }
    joblib.dump(artifact, MODEL)

    contract = {
        "test_roc_auc_min": 0.75,
        "test_balanced_accuracy_min": 0.68,
        "test_recall_min": 0.70,
        "test_green_npv_min": 0.88,
        "red_precision_min_if_red_exists": 0.60,
    }
    red_ok = test_red_metrics["tp"] + test_red_metrics["fp"] == 0 or test_red_metrics["precision"] >= contract["red_precision_min_if_red_exists"]
    gates = {
        "source_rows_exact_1049042": source_rows == 1_049_042,
        "future_actual_sama_not_features": True,
        "chronological_split_with_7day_gaps": TRAIN_END + pd.Timedelta(days=7) < VAL_START and VAL_END + pd.Timedelta(days=7) < TEST_START,
        "threshold_selected_validation_only": True,
        "test_not_used_for_model_or_threshold_selection": True,
        "test_roc_auc": test_metrics["roc_auc"] is not None and test_metrics["roc_auc"] >= contract["test_roc_auc_min"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"] >= contract["test_balanced_accuracy_min"],
        "test_recall": test_metrics["recall"] >= contract["test_recall_min"],
        "test_green_npv": test_metrics["green_npv"] >= contract["test_green_npv_min"],
        "red_precision": bool(red_ok),
    }
    all_passed = bool(all(gates.values()))

    report = {
        "version": VERSION,
        "scientific_boundary": "The 1,049,042 transaction rows originate from UCI Online Retail II and are Saudi-localized/calibrated; they are not observed Saudi merchant transactions. SAMA contributes official aggregate market data and leakage-safe forecast signals.",
        "source_rows": source_rows,
        "eligible_rows": eligible_rows,
        "panel_rows_before_features": int(len(panel)),
        "supervised_rows": int(len(meta)),
        "categories": int(meta.category.nunique()),
        "feature_count": int(X.shape[1]),
        "split": {
            "train_rows": int(tr.sum()), "train_start": str(meta.loc[tr, "date"].min().date()), "train_end": str(meta.loc[tr, "date"].max().date()), "train_positive_rate": float(ytr.mean()),
            "validation_rows": int(va.sum()), "validation_start": str(meta.loc[va, "date"].min().date()), "validation_end": str(meta.loc[va, "date"].max().date()), "validation_positive_rate": float(yva.mean()),
            "test_rows": int(te.sum()), "test_start": str(meta.loc[te, "date"].min().date()), "test_end": str(meta.loc[te, "date"].max().date()), "test_positive_rate": float(yte.mean()),
        },
        "validation_model_auc": candidates,
        "selected": {
            "blend_weight_regression": float(blend_w),
            "watch_threshold": float(watch_t),
            "red_threshold": float(red_t),
            "validation_metrics": val_metrics,
            "validation_red_metrics": val_red_metrics,
            "feasible_watch_thresholds": int(feasible_thresholds),
            "feasible_red_thresholds": int(red_candidates),
        },
        "held_out_test": test_metrics,
        "held_out_test_red": test_red_metrics,
        "contract": contract,
        "gates": gates,
        "all_gates_passed": all_passed,
        "leakage_controls": {
            "future7_target_not_feature": True,
            "actual_future_sama_columns_excluded": True,
            "sama_forecast_origin_is_previous_completed_week": True,
            "test_untouched_until_final_evaluation": True,
            "entity_rows_split_by_global_date_not_randomly": True,
        },
        "artifacts": {"model": str(MODEL.relative_to(ROOT)), "feature_panel": str(FEATURE_PANEL.relative_to(ROOT))},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    SUMMARY.write_text(
        "# Sales Sentinel v4 — Merchant + SAMA Hybrid\n\n"
        f"- Full localized source rows: **{source_rows:,}**\n"
        f"- Categories: **{meta.category.nunique()}**\n"
        f"- Supervised category-day rows: **{len(meta):,}**\n"
        f"- Features: **{X.shape[1]}**\n"
        f"- Held-out Accuracy: **{test_metrics['accuracy']:.2%}**\n"
        f"- Held-out Balanced Accuracy: **{test_metrics['balanced_accuracy']:.2%}**\n"
        f"- Held-out Precision: **{test_metrics['precision']:.2%}**\n"
        f"- Held-out Recall: **{test_metrics['recall']:.2%}**\n"
        f"- Held-out F1: **{test_metrics['f1']:.2%}**\n"
        f"- Held-out ROC-AUC: **{test_metrics['roc_auc']:.2%}**\n"
        f"- GREEN NPV: **{test_metrics['green_npv']:.2%}**\n"
        f"- RED Precision: **{test_red_metrics['precision']:.2%}** ({test_red_metrics['tp'] + test_red_metrics['fp']} RED alerts)\n"
        f"- All scientific gates: **{all_passed}**\n\n"
        "Scientific boundary: row-level merchant transactions are Saudi-localized synthetic microdata derived from UCI Online Retail II; official SAMA data is aggregate market calibration/forecast context, not observed store transactions.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
