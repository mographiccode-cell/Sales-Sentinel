from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PANEL = ROOT / "data" / "saudi_v1_5" / "saudi_sector_daily_panel_v1_5.csv.gz"
REDSEA_FILE = Path(os.environ.get("REDSEA_FILE", "/tmp/redsea_mendeley/RedSea_Data_Cleaned.xlsx"))
OUT = ROOT / "reports" / "redsea_frozen_transfer_v15"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "transfer_report.json"
SUMMARY = OUT / "summary.md"
DEV_OOF = OUT / "development_oof.csv"
REDSEA_PRED = OUT / "redsea_predictions.csv"
MODEL_COMPARISON = OUT / "development_model_comparison.csv"

VERSION = "SALES-SENTINEL-V15-FROZEN-GENERIC-TRANSFER"
DECLINE_THRESHOLD = 0.85
HORIZON = 7
BASELINE_DAYS = 28
RANDOM_STATE = 20260816

RAMADAN = [
    ("2023-03-23", "2023-04-20"),
    ("2024-03-11", "2024-04-09"),
]
EID_FITR = [
    ("2023-04-21", "2023-04-23"),
    ("2024-04-10", "2024-04-12"),
]
HAJJ = [
    ("2023-06-19", "2023-06-30"),
    ("2024-06-07", "2024-06-19"),
]
EID_ADHA = [
    ("2023-06-28", "2023-07-01"),
    ("2024-06-16", "2024-06-19"),
]


def in_ranges(dt: pd.Series, ranges) -> np.ndarray:
    out = np.zeros(len(dt), dtype=int)
    for a, b in ranges:
        out |= dt.between(pd.Timestamp(a), pd.Timestamp(b)).to_numpy(dtype=bool)
    return out.astype(int)


def build_localized_daily() -> pd.DataFrame:
    p = pd.read_csv(SOURCE_PANEL, compression="gzip", parse_dates=["TrainingSafeDate"])
    needed = {"TrainingSafeDate", "sales", "invoices", "customers", "products"}
    missing = needed - set(p.columns)
    if missing:
        raise RuntimeError(f"localized sector panel missing columns: {sorted(missing)}")
    # Counts are sector-summed rather than raw global nunique counts. Every count-derived
    # model feature below is normalized within the same daily series, so stable sector
    # duplication does not create a raw scale advantage.
    d = (
        p.groupby("TrainingSafeDate", as_index=False)
        .agg(
            sales=("sales", "sum"),
            invoices=("invoices", "sum"),
            customers=("customers", "sum"),
            products=("products", "sum"),
        )
        .rename(columns={"TrainingSafeDate": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    full = pd.date_range(d.date.min(), d.date.max(), freq="D")
    d = d.set_index("date").reindex(full).fillna(0.0).rename_axis("date").reset_index()
    d["source"] = "localized_development"
    return d


def build_redsea_daily() -> pd.DataFrame:
    d = pd.read_excel(REDSEA_FILE)
    d = d.drop_duplicates().copy()
    d["TRX DATE"] = pd.to_datetime(d["TRX DATE"], errors="coerce").dt.normalize()
    for c in ["Net Amount", "QUANTITY"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if d["TRX DATE"].isna().any() or d["Net Amount"].isna().any():
        raise RuntimeError("Redsea has invalid TRX DATE or Net Amount after parsing")
    g = (
        d.groupby("TRX DATE", as_index=False)
        .agg(
            sales=("Net Amount", "sum"),
            invoices=("TRX NUMBER", "nunique"),
            customers=("CUSTOMER NUMBER", "nunique"),
            products=("ITEM CODE", "nunique"),
        )
        .rename(columns={"TRX DATE": "date"})
        .sort_values("date")
    )
    full = pd.date_range(g.date.min(), g.date.max(), freq="D")
    g = g.set_index("date").reindex(full).fillna(0.0).rename_axis("date").reset_index()
    g["source"] = "redsea_external_real_saudi"
    return g


def safe_div(a, b):
    a = pd.Series(a, copy=False).astype(float)
    b = pd.Series(b, copy=False).astype(float).replace(0, np.nan)
    return a / b


def add_generic_features(daily: pd.DataFrame):
    d = daily.copy().sort_values("date").reset_index(drop=True)
    dt = pd.to_datetime(d.date)
    X = pd.DataFrame(index=d.index)

    for col in ["sales", "invoices", "customers", "products"]:
        s = pd.to_numeric(d[col], errors="coerce").astype(float)
        ma7 = s.rolling(7, min_periods=7).mean()
        ma14 = s.rolling(14, min_periods=14).mean()
        ma28 = s.rolling(28, min_periods=28).mean()
        sd7 = s.rolling(7, min_periods=7).std()
        sd28 = s.rolling(28, min_periods=28).std()

        X[f"{col}_t0_to_ma28"] = safe_div(s, ma28)
        X[f"{col}_ma7_to_ma28"] = safe_div(ma7, ma28)
        X[f"{col}_ma14_to_ma28"] = safe_div(ma14, ma28)
        X[f"{col}_lag1_to_ma28"] = safe_div(s.shift(1), ma28)
        X[f"{col}_lag7_to_ma28"] = safe_div(s.shift(7), ma28)
        X[f"{col}_lag14_to_ma28"] = safe_div(s.shift(14), ma28)
        X[f"{col}_cv7"] = safe_div(sd7, ma7.abs())
        X[f"{col}_cv28"] = safe_div(sd28, ma28.abs())
        X[f"{col}_chg1"] = safe_div(s, s.shift(1)) - 1.0
        X[f"{col}_chg7"] = safe_div(s, s.shift(7)) - 1.0
        X[f"{col}_trend7_28"] = safe_div(ma7, ma28) - 1.0

    # Scale-free commercial intensity measures.
    derived = pd.DataFrame(index=d.index)
    derived["avg_ticket"] = safe_div(d.sales, d.invoices)
    derived["sales_per_customer"] = safe_div(d.sales, d.customers)
    derived["products_per_invoice"] = safe_div(d.products, d.invoices)
    for col in derived.columns:
        s = derived[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ma7 = s.rolling(7, min_periods=7).mean()
        ma28 = s.rolling(28, min_periods=28).mean()
        X[f"{col}_t0_to_ma28"] = safe_div(s, ma28)
        X[f"{col}_ma7_to_ma28"] = safe_div(ma7, ma28)
        X[f"{col}_chg7"] = safe_div(s, s.shift(7)) - 1.0

    dow = dt.dt.dayofweek.astype(float)
    month = dt.dt.month.astype(float)
    doy = dt.dt.dayofyear.astype(float)
    X["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    X["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    X["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    X["year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["weekend_saudi"] = dt.dt.dayofweek.isin([4, 5]).astype(int)
    X["salary_window"] = dt.dt.day.between(25, 28).astype(int)
    X["founding_day_window"] = ((dt.dt.month == 2) & dt.dt.day.between(20, 24)).astype(int)
    X["national_day_window"] = ((dt.dt.month == 9) & dt.dt.day.between(20, 25)).astype(int)
    X["ramadan"] = in_ranges(dt, RAMADAN)
    X["eid_fitr"] = in_ranges(dt, EID_FITR)
    X["hajj"] = in_ranges(dt, HAJJ)
    X["eid_adha"] = in_ranges(dt, EID_ADHA)

    baseline = pd.to_numeric(d.sales, errors="coerce").rolling(BASELINE_DAYS, min_periods=BASELINE_DAYS).mean()
    future = sum(pd.to_numeric(d.sales, errors="coerce").shift(-k) for k in range(1, HORIZON + 1))
    future_ratio = future / (HORIZON * baseline.replace(0, np.nan))
    target = (future_ratio < DECLINE_THRESHOLD).astype(int)
    future_complete = pd.concat([d.sales.shift(-k) for k in range(1, HORIZON + 1)], axis=1).notna().all(axis=1)

    X = X.replace([np.inf, -np.inf], np.nan)
    good = X.notna().all(axis=1) & future_ratio.notna() & future_complete
    meta = d.loc[good, ["date", "sales", "invoices", "customers", "products", "source"]].reset_index(drop=True)
    meta["future_ratio"] = future_ratio.loc[good].to_numpy(float)
    meta["target"] = target.loc[good].to_numpy(int)
    return meta, X.loc[good].reset_index(drop=True)


def model_candidates():
    out = []
    for c in [0.1, 0.3, 1.0, 3.0]:
        out.append((
            f"logistic_c{c}",
            Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=c, class_weight="balanced", max_iter=4000, random_state=RANDOM_STATE)),
            ]),
        ))
    for leaf in [3, 5, 8]:
        for mf in [0.5, 0.8]:
            out.append((
                f"extra_leaf{leaf}_mf{mf}",
                ExtraTreesClassifier(
                    n_estimators=450,
                    min_samples_leaf=leaf,
                    max_features=mf,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ))
    for l2 in [1.0, 5.0, 10.0]:
        out.append((
            f"hist_l2_{l2}",
            HistGradientBoostingClassifier(
                learning_rate=0.04,
                max_iter=250,
                max_leaf_nodes=7,
                min_samples_leaf=15,
                l2_regularization=l2,
                random_state=RANDOM_STATE,
            ),
        ))
    return out


def positive_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))


def safe_rank_metrics(y, score):
    if len(np.unique(y)) < 2:
        return None, None
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def threshold_metrics(y, score, threshold):
    p = np.asarray(score) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "npv": float(tn / (tn + fn)) if (tn + fn) else 0.0,
        "alert_rate": float(p.mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def choose_threshold(y, score):
    rows = []
    # Threshold search happens on development OOF only.
    for t in np.linspace(0.05, 0.95, 181):
        m = threshold_metrics(y, score, t)
        rows.append(m)
    feasible = [m for m in rows if m["recall"] >= 0.80 and m["alert_rate"] <= 0.45]
    if not feasible:
        feasible = [m for m in rows if m["recall"] >= 0.80]
    if not feasible:
        feasible = rows
    return max(feasible, key=lambda m: (m["f1"], m["precision"], -m["alert_rate"]))


def clone_model(name):
    for n, m in model_candidates():
        if n == name:
            return m
    raise KeyError(name)


def bootstrap_ci(y, score, threshold, n_boot=2000):
    rng = np.random.default_rng(RANDOM_STATE)
    y = np.asarray(y, int); score = np.asarray(score, float)
    vals = {"roc_auc": [], "pr_auc": [], "precision": [], "recall": [], "f1": []}
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]; ss = score[idx]
        if len(np.unique(yy)) < 2:
            continue
        auc, pr = safe_rank_metrics(yy, ss)
        mm = threshold_metrics(yy, ss, threshold)
        vals["roc_auc"].append(auc); vals["pr_auc"].append(pr)
        vals["precision"].append(mm["precision"]); vals["recall"].append(mm["recall"]); vals["f1"].append(mm["f1"])
    return {
        k: {"low": float(np.quantile(v, 0.025)), "high": float(np.quantile(v, 0.975)), "n": len(v)}
        for k, v in vals.items() if v
    }


def main():
    dev_daily = build_localized_daily()
    ext_daily = build_redsea_daily()
    dev_meta, Xdev = add_generic_features(dev_daily)
    ext_meta, Xext = add_generic_features(ext_daily)
    if list(Xdev.columns) != list(Xext.columns):
        raise RuntimeError("development/external generic feature schema mismatch")

    y = dev_meta.target.to_numpy(int)
    n = len(y)
    # Five purged chronological validation windows. The 7-row gap prevents training labels
    # from overlapping the first validation target horizon.
    test_size = min(70, max(30, (n - 120) // 5))
    cv = TimeSeriesSplit(n_splits=5, test_size=test_size, gap=HORIZON)

    comparison = []
    all_scores = {}
    all_folds = np.full(n, -1, dtype=int)
    splits = list(cv.split(Xdev))
    for fold, (_, va) in enumerate(splits):
        all_folds[va] = fold

    for name, _ in model_candidates():
        oof = np.full(n, np.nan, dtype=float)
        per_fold = []
        for fold, (tr, va) in enumerate(splits):
            model = clone_model(name)
            model.fit(Xdev.iloc[tr], y[tr])
            s = positive_score(model, Xdev.iloc[va])
            oof[va] = s
            auc, pr = safe_rank_metrics(y[va], s)
            per_fold.append({"fold": fold, "train_rows": len(tr), "validation_rows": len(va), "roc_auc": auc, "pr_auc": pr, "positive_rate": float(y[va].mean())})
        mask = np.isfinite(oof)
        auc, pr = safe_rank_metrics(y[mask], oof[mask])
        thr = choose_threshold(y[mask], oof[mask])
        comparison.append({"model": name, "roc_auc": auc, "pr_auc": pr, **{f"op_{k}": v for k, v in thr.items()}, "per_fold": per_fold})
        all_scores[name] = oof

    # Rank by PR-AUC first because the development decline target is imbalanced; AUC is tie-breaker.
    selected = max(comparison, key=lambda r: (r["pr_auc"], r["roc_auc"], r["op_f1"]))
    selected_name = selected["model"]
    threshold = float(selected["op_threshold"])
    oof = all_scores[selected_name]
    mask = np.isfinite(oof)
    dev_auc, dev_pr = safe_rank_metrics(y[mask], oof[mask])
    dev_operating = threshold_metrics(y[mask], oof[mask], threshold)

    oof_df = dev_meta.copy()
    oof_df["fold_id"] = all_folds
    oof_df["score"] = oof
    oof_df["prediction"] = np.where(np.isfinite(oof), (oof >= threshold).astype(int), -1)
    oof_df.to_csv(DEV_OOF, index=False)

    pd.DataFrame([{k: v for k, v in r.items() if k != "per_fold"} for r in comparison]).sort_values(["pr_auc", "roc_auc"], ascending=False).to_csv(MODEL_COMPARISON, index=False)

    # Freeze model and threshold using development data only, then open Redsea once.
    final_model = clone_model(selected_name)
    final_model.fit(Xdev, y)
    y_ext = ext_meta.target.to_numpy(int)
    ext_score = positive_score(final_model, Xext)
    ext_auc, ext_pr = safe_rank_metrics(y_ext, ext_score)
    ext_operating = threshold_metrics(y_ext, ext_score, threshold)
    ci = bootstrap_ci(y_ext, ext_score, threshold)

    ext_pred = ext_meta.copy()
    ext_pred["score"] = ext_score
    ext_pred["prediction"] = (ext_score >= threshold).astype(int)
    ext_pred.to_csv(REDSEA_PRED, index=False)

    report = {
        "version": VERSION,
        "status": "FROZEN_TRANSFER_EVALUATED",
        "scientific_boundary": (
            "Model family, feature schema, and decision threshold were selected using only the localized development series. "
            "Redsea labels were not used for fitting, feature selection, model selection, probability calibration, or threshold selection. "
            "The localized development microdata remain UCI-derived and Saudi-calibrated; Redsea is the independent real Saudi merchant dataset."
        ),
        "target": {"definition": "next 7-day sales / (7 * trailing 28-day daily mean including prediction day) < 0.85", "decline": "15%+"},
        "features": {"count": int(Xdev.shape[1]), "scale_free": True, "columns": list(Xdev.columns)},
        "development": {
            "daily_rows": int(len(dev_daily)), "eligible_rows": int(len(dev_meta)), "oof_rows": int(mask.sum()),
            "date_start": str(dev_meta.date.min().date()), "date_end": str(dev_meta.date.max().date()),
            "positive_rate": float(dev_meta.target.mean()), "selected_model": selected_name, "frozen_threshold": threshold,
            "roc_auc": dev_auc, "pr_auc": dev_pr, "operating": dev_operating,
            "selection_rule": "highest development OOF PR-AUC, then ROC-AUC, then F1; threshold optimized on development OOF with recall>=80% and alert<=45% when feasible",
        },
        "redsea_external": {
            "source": "Mendeley Data Redsea Dataset DOI 10.17632/9c87bd42ct.1",
            "sha256": "dd994d25042e54119babe058820b02e00b3443fe18150b9213f9c6929466a645",
            "daily_rows": int(len(ext_daily)), "eligible_rows": int(len(ext_meta)),
            "date_start": str(ext_meta.date.min().date()), "date_end": str(ext_meta.date.max().date()),
            "positive_rate": float(ext_meta.target.mean()), "roc_auc": ext_auc, "pr_auc": ext_pr,
            "operating_at_frozen_development_threshold": ext_operating,
            "bootstrap_95pct_ci": ci,
            "simple_recent7_vs28_reference_auc_from_prior_audit": 0.4230,
        },
        "development_models": comparison,
        "red_supported": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    dm = report["development"]; ex = report["redsea_external"]; op = ex["operating_at_frozen_development_threshold"]
    lines = [
        "# Sales Sentinel V15 — Frozen Generic Transfer to Redsea", "",
        f"- Status: **{report['status']}**",
        f"- Generic scale-free features: **{report['features']['count']}**",
        f"- Development model: **{dm['selected_model']}**",
        f"- Frozen development threshold: **{dm['frozen_threshold']:.3f}**", "",
        "## Development-only OOF",
        f"- Rows / decline prevalence: **{dm['oof_rows']} / {dm['positive_rate']:.2%}**",
        f"- ROC-AUC / PR-AUC: **{dm['roc_auc']:.2%} / {dm['pr_auc']:.2%}**",
        f"- Precision / Recall / F1: **{dm['operating']['precision']:.2%} / {dm['operating']['recall']:.2%} / {dm['operating']['f1']:.2%}**",
        f"- TP/FP/FN/TN: **{dm['operating']['tp']}/{dm['operating']['fp']}/{dm['operating']['fn']}/{dm['operating']['tn']}**", "",
        "## Independent real Saudi merchant: Redsea",
        f"- Eligible dates / decline prevalence: **{ex['eligible_rows']} / {ex['positive_rate']:.2%}**",
        f"- ROC-AUC / PR-AUC: **{ex['roc_auc']:.2%} / {ex['pr_auc']:.2%}**",
        f"- Precision / Recall / F1 at frozen threshold: **{op['precision']:.2%} / {op['recall']:.2%} / {op['f1']:.2%}**",
        f"- Accuracy / Balanced Accuracy: **{op['accuracy']:.2%} / {op['balanced_accuracy']:.2%}**",
        f"- NPV / Alert rate: **{op['npv']:.2%} / {op['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{op['tp']}/{op['fp']}/{op['fn']}/{op['tn']}**",
        f"- Prior simple recent7/28 Redsea AUC: **{ex['simple_recent7_vs28_reference_auc_from_prior_audit']:.2%}**", "",
        "Important: Redsea was opened only after the generic feature schema, model family selection rule, and threshold were frozen on development data. No Redsea label was used to improve these reported external metrics.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
