from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

import train_merchant_category_signals_v7_1 as v71

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DAILY = ROOT / "data" / "saudi_v1_3" / "saudi_daily_sama_calibrated_v1_3.csv"
SAMA_WEEKLY = ROOT / "data" / "saudi_v1_3" / "saudi_weekly_sama_calibration_v1_3.csv"
REDSEA_FILE = Path(os.environ.get("REDSEA_FILE", "/tmp/redsea_mendeley/RedSea_Data_Cleaned.xlsx"))
V71_REPORT = ROOT / "reports" / "merchant_category_signals_v7_1" / "development_report.json"
V15_REPORT = ROOT / "reports" / "redsea_frozen_transfer_v15" / "transfer_report.json"
OUT = ROOT / "reports" / "redsea_portable_v16"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "diagnostic_report.json"
SUMMARY = OUT / "summary.md"
DEV_OOF = OUT / "development_oof.csv"
REDSEA_PRED = OUT / "redsea_predictions.csv"
DRIFT = OUT / "feature_drift.csv"

VERSION = "SALES-SENTINEL-V16-PORTABLE-V7.1-EXTERNAL-DIAGNOSTIC"
SEED = 20260816


def week_start_sunday(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s).dt.normalize()
    return x - pd.to_timedelta((x.dt.dayofweek + 1) % 7, unit="D")


def load_prior_week_market(dates: pd.Series) -> np.ndarray:
    w = pd.read_csv(SAMA_WEEKLY, parse_dates=["SAMAWeekStart"])[["SAMAWeekStart", "SAMAWeeklyMarketIndex"]]
    w = w.drop_duplicates("SAMAWeekStart").sort_values("SAMAWeekStart")
    lookup = dict(zip(w.SAMAWeekStart.dt.normalize(), pd.to_numeric(w.SAMAWeeklyMarketIndex, errors="coerce")))
    cur = week_start_sunday(pd.Series(pd.to_datetime(dates)))
    # Strictly previous completed week only.
    return np.asarray([lookup.get(x - pd.Timedelta(days=7), np.nan) for x in cur], dtype=float)


def source_daily() -> pd.DataFrame:
    d = pd.read_csv(SOURCE_DAILY, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    keep = [
        "date", "sama_calibrated_net_sales_sar", "gross_sales_sar", "invoice_count",
        "unique_observed_customers", "new_observed_customers", "returning_observed_customers",
        "unique_products", "units", "average_invoice_value_sar", "return_rate_value", "transaction_rows",
    ]
    d = d[keep].copy()
    d["sama_weekly_market_index"] = load_prior_week_market(d.date)
    return d


def redsea_daily() -> pd.DataFrame:
    raw = pd.read_excel(REDSEA_FILE).drop_duplicates().copy()
    raw["date"] = pd.to_datetime(raw["TRX DATE"], errors="coerce").dt.normalize()
    for c in ["Net Amount", "QUANTITY"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    if raw.date.isna().any() or raw["Net Amount"].isna().any():
        raise RuntimeError("Redsea date/Net Amount parse failure")

    # First-seen/returning customer status is defined only within the observed Redsea window.
    first = raw.groupby("CUSTOMER NUMBER", dropna=False).date.min().rename("first_date")
    z = raw.merge(first, left_on="CUSTOMER NUMBER", right_index=True, how="left")
    z["is_new_customer"] = z.date.eq(z.first_date)
    z["is_returning_customer"] = z.date.gt(z.first_date)
    z["positive_net"] = z["Net Amount"].clip(lower=0)
    z["return_value"] = -z["Net Amount"].clip(upper=0)
    z["abs_units"] = z.QUANTITY.abs()

    rows = []
    for day, g in z.groupby("date", sort=True):
        inv = int(g["TRX NUMBER"].nunique())
        sales = float(g["Net Amount"].sum())
        customers = int(g["CUSTOMER NUMBER"].nunique())
        new_customers = int(g.loc[g.is_new_customer, "CUSTOMER NUMBER"].nunique())
        returning = int(g.loc[g.is_returning_customer, "CUSTOMER NUMBER"].nunique())
        gross = float(g.positive_net.sum())
        returns = float(g.return_value.sum())
        rows.append({
            "date": day,
            "sama_calibrated_net_sales_sar": sales,
            "gross_sales_sar": gross,
            "invoice_count": inv,
            "unique_observed_customers": customers,
            "new_observed_customers": new_customers,
            "returning_observed_customers": returning,
            "unique_products": int(g["ITEM CODE"].nunique()),
            "units": float(g.abs_units.sum()),
            "average_invoice_value_sar": sales / max(inv, 1),
            "return_rate_value": returns / max(gross, 1e-9),
            "transaction_rows": int(len(g)),
        })
    d = pd.DataFrame(rows).sort_values("date")
    full = pd.date_range(d.date.min(), d.date.max(), freq="D")
    d = d.set_index("date").reindex(full).rename_axis("date").reset_index()
    numeric = [c for c in d.columns if c != "date"]
    d[numeric] = d[numeric].fillna(0.0)
    d["sama_weekly_market_index"] = load_prior_week_market(d.date)
    return d


def build_meta_and_features(d: pd.DataFrame):
    d = d.copy().sort_values("date").reset_index(drop=True)
    X = v71.merchant_features(d)
    # Portable features must be reproducible from both domains and avoid raw merchant scale.
    portable = [
        c for c in X.columns
        if not c.endswith("__log")
        and c != "merchant__electronic_share"
    ]
    X = X[portable].replace([np.inf, -np.inf], np.nan)

    sales = pd.to_numeric(d["sama_calibrated_net_sales_sar"], errors="coerce").clip(lower=0).astype(float)
    baseline28 = sales.rolling(28, min_periods=28).mean()
    future7 = sum(sales.shift(-h) for h in range(1, 8))
    ratio = future7 / (7.0 * baseline28.replace(0, np.nan))
    target = (ratio < 0.85).astype(int)
    good = (d.date >= d.date.min() + pd.Timedelta(days=56)) & ratio.notna() & baseline28.gt(0)
    meta = pd.DataFrame({"date": d.date, "future_ratio": ratio, "target": target}).loc[good].reset_index(drop=True)
    X = X.loc[good].reset_index(drop=True)
    return meta, X, portable


def score_model(model, X):
    return model.predict_proba(X)[:, 1]


def fit_full(selected_model: str, Xdev: pd.DataFrame, ydev: pd.Series, Xext: pd.DataFrame):
    Xfit, Xout, prep = v71.fold_prepare(Xdev, Xext)
    if selected_model == "mean_ensemble":
        models = {name: v71.fit_model(clone(factory), Xfit, ydev) for name, factory in v71.factories().items()}
        score = np.column_stack([score_model(m, Xout) for m in models.values()]).mean(axis=1)
        fitted_kind = "mean_ensemble"
    else:
        model = v71.fit_model(clone(v71.factories()[selected_model]), Xfit, ydev)
        score = score_model(model, Xout)
        fitted_kind = selected_model
    return score, prep, fitted_kind, Xfit, Xout


def bootstrap_ci(y, score, threshold, n_boot=3000):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y, int); score = np.asarray(score, float); n = len(y)
    vals = {"roc_auc": [], "pr_auc": [], "precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n); yy = y[idx]; ss = score[idx]
        if len(np.unique(yy)) < 2:
            continue
        m = v71.metrics(yy, ss, threshold)
        vals["roc_auc"].append(float(roc_auc_score(yy, ss)))
        vals["pr_auc"].append(float(average_precision_score(yy, ss)))
        vals["precision"].append(m["precision"]); vals["recall"].append(m["recall"]); vals["f1"].append(m["f1"])
    return {k: {"low": float(np.quantile(v, .025)), "high": float(np.quantile(v, .975)), "n": len(v)} for k, v in vals.items()}


def drift_table(Xfit: pd.DataFrame, Xext: pd.DataFrame, prep: dict) -> pd.DataFrame:
    rows = []
    for c in Xfit.columns:
        a = pd.to_numeric(Xfit[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(Xext[c], errors="coerce").to_numpy(float)
        pooled = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2.0)
        smd = float((np.nanmean(b) - np.nanmean(a)) / pooled) if pooled > 1e-12 else 0.0
        p = prep[c]
        raw_ext = b
        clip_share = float(np.mean((raw_ext <= p["p01"] + 1e-12) | (raw_ext >= p["p99"] - 1e-12)))
        rows.append({"feature": c, "standardized_mean_difference": smd, "abs_smd": abs(smd), "external_at_source_clip_boundary_share": clip_share})
    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False)


def main():
    dev_daily = source_daily()
    ext_daily = redsea_daily()
    dev_meta, Xdev, cols = build_meta_and_features(dev_daily)
    ext_meta, Xext, ext_cols = build_meta_and_features(ext_daily)
    if cols != ext_cols or list(Xdev.columns) != list(Xext.columns):
        raise RuntimeError("Portable feature schema mismatch")
    if len(dev_meta) != 541:
        raise RuntimeError(f"Expected 541 development rows, got {len(dev_meta)}")

    run, oof = v71.nested_run(dev_meta, Xdev, cols, "portable_merchant")
    choices = [("portable_merchant", name, result) for name, result in run["models"].items()]
    _, selected_model, selected_result = max(choices, key=v71.selection_key)
    threshold = float(selected_result["median_inner_threshold"])
    dev_metrics = selected_result["nested_oof_metrics"]

    selected_oof = oof[oof.model.eq(selected_model)].sort_values(["fold_id", "date"]).copy()
    selected_oof.to_csv(DEV_OOF, index=False)

    ext_score, prep, fitted_kind, Xfit, Xout = fit_full(selected_model, Xdev, dev_meta.target.astype(int), Xext)
    yext = ext_meta.target.to_numpy(int)
    ext_metrics = v71.metrics(yext, ext_score, threshold)
    ext_pred = ext_meta.copy(); ext_pred["score"] = ext_score; ext_pred["prediction"] = (ext_score >= threshold).astype(int)
    ext_pred.to_csv(REDSEA_PRED, index=False)

    dr = drift_table(Xfit, Xout, prep)
    dr.to_csv(DRIFT, index=False)

    v71_full = json.loads(V71_REPORT.read_text(encoding="utf-8")) if V71_REPORT.exists() else None
    v15 = json.loads(V15_REPORT.read_text(encoding="utf-8")) if V15_REPORT.exists() else None
    report = {
        "version": VERSION,
        "status": "POST_OPEN_EXTERNAL_DIAGNOSTIC",
        "scientific_boundary": (
            "Redsea was already opened in V15, therefore V16 is explicitly a post-open external diagnostic and must not be called a new blind validation. "
            "However, V16 model/scope selection still uses only nested rolling-origin development data; Redsea labels do not choose the model or threshold."
        ),
        "portable_feature_count": len(cols),
        "portable_feature_columns": cols,
        "development": {
            "rows": len(dev_meta), "positive_rate": float(dev_meta.target.mean()), "selected_model": selected_model,
            "threshold": threshold, "nested_oof_metrics": dev_metrics,
            "worst_fold_recall": selected_result["worst_fold_recall"], "max_fold_alert_rate": selected_result["max_fold_alert_rate"],
        },
        "redsea": {
            "source": "Mendeley Redsea Dataset DOI 10.17632/9c87bd42ct.1",
            "sha256": "dd994d25042e54119babe058820b02e00b3443fe18150b9213f9c6929466a645",
            "eligible_rows": len(ext_meta), "positive_rate": float(ext_meta.target.mean()),
            "date_start": str(ext_meta.date.min().date()), "date_end": str(ext_meta.date.max().date()),
            "metrics_at_development_frozen_threshold": ext_metrics,
            "bootstrap_95pct_ci": bootstrap_ci(yext, ext_score, threshold),
        },
        "drift": {
            "median_abs_smd": float(dr.abs_smd.median()),
            "max_abs_smd": float(dr.abs_smd.max()),
            "features_abs_smd_ge_1": int((dr.abs_smd >= 1.0).sum()),
            "median_external_clip_boundary_share": float(dr.external_at_source_clip_boundary_share.median()),
            "top_10": dr.head(10).to_dict("records"),
        },
        "references": {
            "v7_1_full_nested_auc": None if not v71_full else v71_full["selected"]["nested_oof_metrics"]["roc_auc"],
            "v7_1_full_nested_pr_auc": None if not v71_full else v71_full["selected"]["nested_oof_metrics"]["pr_auc"],
            "v15_redsea_auc": None if not v15 else v15["redsea_external"]["roc_auc"],
        },
        "red_supported": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    dm = report["development"]; em = report["redsea"]["metrics_at_development_frozen_threshold"]; drift = report["drift"]
    lines = [
        "# Sales Sentinel V16 — Portable V7.1 → Redsea Diagnostic", "",
        f"- Status: **{report['status']}**",
        f"- Portable V7.1 features: **{report['portable_feature_count']}**",
        f"- Selected model from nested development only: **{dm['selected_model']}**",
        f"- Frozen median-inner threshold: **{dm['threshold']:.3f}**", "",
        "## Nested development",
        f"- Rows / decline prevalence: **{dm['rows']} / {dm['positive_rate']:.2%}**",
        f"- ROC-AUC / PR-AUC: **{dm['nested_oof_metrics']['roc_auc']:.2%} / {dm['nested_oof_metrics']['pr_auc']:.2%}**",
        f"- Precision / Recall / F1: **{dm['nested_oof_metrics']['precision']:.2%} / {dm['nested_oof_metrics']['recall']:.2%} / {dm['nested_oof_metrics']['f1']:.2%}**",
        f"- NPV / Alert rate: **{dm['nested_oof_metrics']['green_npv']:.2%} / {dm['nested_oof_metrics']['alert_rate']:.2%}**", "",
        "## Redsea post-open external diagnostic",
        f"- Eligible rows / decline prevalence: **{report['redsea']['eligible_rows']} / {report['redsea']['positive_rate']:.2%}**",
        f"- ROC-AUC / PR-AUC: **{em['roc_auc']:.2%} / {em['pr_auc']:.2%}**",
        f"- Precision / Recall / F1: **{em['precision']:.2%} / {em['recall']:.2%} / {em['f1']:.2%}**",
        f"- Accuracy / Balanced Accuracy: **{em['accuracy']:.2%} / {em['balanced_accuracy']:.2%}**",
        f"- NPV / Alert rate: **{em['green_npv']:.2%} / {em['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{em['tp']}/{em['fp']}/{em['fn']}/{em['tn']}**", "",
        "## Domain drift",
        f"- Median |SMD|: **{drift['median_abs_smd']:.3f}**",
        f"- Max |SMD|: **{drift['max_abs_smd']:.3f}**",
        f"- Features with |SMD| >= 1: **{drift['features_abs_smd_ge_1']}**", "",
        "Scientific note: V16 is not called blind because Redsea outcomes were already opened in V15. Its purpose is to diagnose whether V7.1's portable merchant feature family transfers better than the weak generic V15 feature set without using Redsea labels for model selection.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
