from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V7.1-MERCHANT-CATEGORY-SIGNALS-NESTED"
SEED = 42
DECLINE = 0.15
PURGE_DAYS = 7

DAILY = ROOT / "data" / "saudi_v1_3" / "saudi_daily_sama_calibrated_v1_3.csv"
CATEGORY = ROOT / "data" / "merchant_v4" / "merchant_sector_daily_features_v4.csv"
V61_REPORT = ROOT / "reports" / "merchant_market_fusion_v6_1" / "development_report.json"

OUT = ROOT / "reports" / "merchant_category_signals_v7_1"
DATA = ROOT / "data" / "merchant_v7_1"
MOD = ROOT / "models" / "merchant_category_signals_v7_1"
for p in (OUT, DATA, MOD):
    p.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_predictions.csv"
FEATURE_MANIFEST = OUT / "feature_manifest.json"
PANEL = DATA / "merchant_feature_panel_v7_1.csv"
MODEL = MOD / "merchant_category_signals_v7_1.joblib"


def safe_change(s: pd.Series, lag: int) -> pd.Series:
    prev = s.shift(lag)
    return (s - prev) / prev.abs().replace(0, np.nan)


def safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(float) / b.astype(float).replace(0, np.nan)


def merchant_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("date").reset_index(drop=True)
    X = pd.DataFrame(index=d.index)
    available = [
        "sama_calibrated_net_sales_sar",
        "gross_sales_sar",
        "invoice_count",
        "unique_observed_customers",
        "new_observed_customers",
        "returning_observed_customers",
        "unique_products",
        "units",
        "average_invoice_value_sar",
        "return_rate_value",
        "transaction_rows",
        "electronic_invoice_count",
    ]
    base = [c for c in available if c in d.columns]
    if "sama_calibrated_net_sales_sar" not in base:
        raise RuntimeError("Missing merchant sales column")

    for c in base:
        s = pd.to_numeric(d[c], errors="coerce").astype(float)
        prefix = c.replace("sama_calibrated_", "")
        X[f"merchant__{prefix}__log"] = np.log1p(s.clip(lower=0))
        for w in (7, 14, 28, 56):
            mean = s.rolling(w, min_periods=w).mean()
            std = s.rolling(w, min_periods=w).std()
            X[f"merchant__{prefix}__ratio_mean_{w}"] = safe_ratio(s, mean)
            if c in {"sama_calibrated_net_sales_sar", "invoice_count", "unique_observed_customers"}:
                X[f"merchant__{prefix}__z_{w}"] = (s - mean) / std.replace(0, np.nan)
        for lag in (1, 7, 14, 28):
            X[f"merchant__{prefix}__change_{lag}"] = safe_change(s, lag)

    sales = pd.to_numeric(d["sama_calibrated_net_sales_sar"], errors="coerce").astype(float)
    X["merchant__sales__ma7_vs_ma28"] = safe_ratio(sales.rolling(7, min_periods=7).mean(), sales.rolling(28, min_periods=28).mean())
    X["merchant__sales__ma14_vs_ma56"] = safe_ratio(sales.rolling(14, min_periods=14).mean(), sales.rolling(56, min_periods=56).mean())
    X["merchant__sales__vol7_vs_vol28"] = safe_ratio(sales.rolling(7, min_periods=7).std(), sales.rolling(28, min_periods=28).std())
    X["merchant__sales__drawdown28"] = safe_ratio(sales, sales.rolling(28, min_periods=28).max()) - 1.0

    if {"electronic_invoice_count", "invoice_count"}.issubset(d.columns):
        X["merchant__electronic_share"] = safe_ratio(d.electronic_invoice_count, d.invoice_count.clip(lower=1))
    if {"new_observed_customers", "unique_observed_customers"}.issubset(d.columns):
        X["merchant__new_customer_share"] = safe_ratio(d.new_observed_customers, d.unique_observed_customers.clip(lower=1))
    if {"returning_observed_customers", "unique_observed_customers"}.issubset(d.columns):
        X["merchant__returning_customer_share"] = safe_ratio(d.returning_observed_customers, d.unique_observed_customers.clip(lower=1))

    for c in ["sama_weekly_market_index", "sama_national_calibration_factor"]:
        if c in d.columns:
            s = pd.to_numeric(d[c], errors="coerce").astype(float)
            X[f"market__{c}"] = s
            X[f"market__{c}__change_1"] = safe_change(s, 1)
            X[f"market__{c}__change_7"] = safe_change(s, 7)

    dates = pd.to_datetime(d.date)
    dow = dates.dt.dayofweek.astype(float)
    doy = dates.dt.dayofyear.astype(float)
    X["calendar__dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    X["calendar__dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    X["calendar__year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["calendar__year_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["calendar__salary_period"] = dates.dt.day.between(24, 31).astype(float)
    X["calendar__national_day_window"] = ((dates.dt.month == 9) & dates.dt.day.between(16, 30)).astype(float)
    X["calendar__founding_day_window"] = ((dates.dt.month == 2) & dates.dt.day.between(15, 29)).astype(float)
    return X


def category_signal_features(c: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    c["date"] = pd.to_datetime(c["date"]).dt.normalize()
    signal_candidates = [
        "net_sales_ratio_mean_7", "net_sales_ratio_mean_14", "net_sales_ratio_mean_28", "net_sales_ratio_mean_56",
        "net_sales_change_1", "net_sales_change_7", "net_sales_change_14", "net_sales_change_28",
        "invoice_count_ratio_mean_7", "invoice_count_ratio_mean_28", "invoice_count_change_7", "invoice_count_change_28",
        "observed_customer_count_ratio_mean_7", "observed_customer_count_ratio_mean_28", "observed_customer_count_change_7",
        "unique_products_ratio_mean_7", "unique_products_ratio_mean_28", "unique_products_change_7",
        "avg_invoice_value_ratio_mean_7", "avg_invoice_value_ratio_mean_28", "avg_invoice_value_change_7",
        "return_rate_value_ratio_mean_7", "return_rate_value_change_7",
        "category_share_ratio_28", "category_share_change_7",
        "sama_predicted_value_h1_change_vs_last", "sama_predicted_value_h2_change_vs_last",
        "sama_predicted_count_h1_change_vs_last", "sama_predicted_count_h2_change_vs_last",
    ]
    signals = [x for x in signal_candidates if x in c.columns]
    if len(signals) < 10:
        raise RuntimeError(f"Too few category signals available: {signals}")

    rows = []
    for day, g in c.groupby("date", sort=True):
        row: dict[str, float | pd.Timestamp] = {"date": day}
        weights = None
        if "category_share" in g.columns:
            w = pd.to_numeric(g["category_share"], errors="coerce").clip(lower=0).fillna(0).to_numpy(float)
            if w.sum() > 0:
                weights = w / w.sum()
        for col in signals:
            x = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
            if len(x) == 0:
                continue
            row[f"catagg__{col}__mean"] = float(np.mean(x))
            row[f"catagg__{col}__std"] = float(np.std(x))
            row[f"catagg__{col}__p10"] = float(np.quantile(x, 0.10))
            row[f"catagg__{col}__p90"] = float(np.quantile(x, 0.90))
            if weights is not None and len(weights) == len(g):
                vals = pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
                ok = np.isfinite(vals)
                if ok.any() and weights[ok].sum() > 0:
                    row[f"catagg__{col}__weighted_mean"] = float(np.average(vals[ok], weights=weights[ok]))

        if "net_sales_ratio_mean_7" in g.columns:
            x = pd.to_numeric(g["net_sales_ratio_mean_7"], errors="coerce").to_numpy(float)
            row["catregime__share_sales_below_090_ma7"] = float(np.nanmean(x < 0.90))
            row["catregime__share_sales_below_100_ma7"] = float(np.nanmean(x < 1.00))
            row["catregime__share_sales_above_110_ma7"] = float(np.nanmean(x > 1.10))
        if "net_sales_change_7" in g.columns:
            x = pd.to_numeric(g["net_sales_change_7"], errors="coerce").to_numpy(float)
            row["catregime__share_negative_sales_change7"] = float(np.nanmean(x < 0.0))
            row["catregime__share_severe_sales_change7"] = float(np.nanmean(x < -0.15))
        if "invoice_count_change_7" in g.columns:
            x = pd.to_numeric(g["invoice_count_change_7"], errors="coerce").to_numpy(float)
            row["catregime__share_negative_invoice_change7"] = float(np.nanmean(x < 0.0))
        if "observed_customer_count_change_7" in g.columns:
            x = pd.to_numeric(g["observed_customer_count_change_7"], errors="coerce").to_numpy(float)
            row["catregime__share_negative_customer_change7"] = float(np.nanmean(x < 0.0))
        if "category_share" in g.columns:
            share = pd.to_numeric(g["category_share"], errors="coerce").clip(lower=0).fillna(0).to_numpy(float)
            if share.sum() > 0:
                share = share / share.sum()
                row["catregime__sales_share_hhi"] = float(np.square(share).sum())
                row["catregime__largest_category_share"] = float(np.max(share))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Cross-category regime changes are merchant-level past-only signals.
    for col in [x for x in out.columns if x.startswith("catregime__")]:
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}__delta7"] = s - s.shift(7)
        out[f"{col}__delta28"] = s - s.shift(28)
    return out


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    d = pd.read_csv(DAILY, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    c = pd.read_csv(CATEGORY, parse_dates=["date"])
    merchant = merchant_features(d)
    category = category_signal_features(c)
    merged = d[["date"]].merge(category, on="date", how="left", validate="one_to_one")
    cat_cols = [x for x in merged.columns if x != "date"]
    X = pd.concat([merchant, merged[cat_cols]], axis=1)

    sales = pd.to_numeric(d["sama_calibrated_net_sales_sar"], errors="coerce").clip(lower=0).astype(float)
    baseline28 = sales.rolling(28, min_periods=28).mean()
    future7 = sum(sales.shift(-h) for h in range(1, 8))
    future_ratio = future7 / (7.0 * baseline28.replace(0, np.nan))
    target = (future_ratio < (1.0 - DECLINE)).astype(int)

    X = X.replace([np.inf, -np.inf], np.nan)
    # Missingness is retained for fold-local imputation; only target validity gates rows.
    good = (d.date >= d.date.min() + pd.Timedelta(days=56)) & future_ratio.notna() & baseline28.gt(0)
    meta = pd.DataFrame({
        "date": d.date,
        "future_ratio": future_ratio,
        "future7_sales": future7,
        "baseline28_daily": baseline28,
        "target": target,
    }).loc[good].reset_index(drop=True)
    X = X.loc[good].reset_index(drop=True)
    merchant_cols = [x for x in X.columns if x.startswith("merchant__") or x.startswith("market__") or x.startswith("calendar__")]
    full_cols = list(X.columns)
    return meta, X, merchant_cols, full_cols


def factories() -> dict[str, object]:
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(C=0.12, class_weight="balanced", max_iter=6000, random_state=SEED)),
        "extra_trees": ExtraTreesClassifier(n_estimators=1000, max_depth=7, min_samples_leaf=6, max_features=0.55, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=900, max_depth=7, min_samples_leaf=6, max_features=0.55, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "hist_gb": HistGradientBoostingClassifier(max_iter=350, learning_rate=0.025, max_leaf_nodes=12, min_samples_leaf=18, l2_regularization=10.0, random_state=SEED),
    }


def fold_prepare(Xtr: pd.DataFrame, Xva: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tr = Xtr.copy()
    va = Xva.copy()
    clip_meta = {}
    for col in tr.columns:
        s = pd.to_numeric(tr[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = s.dropna()
        if finite.empty:
            lo = hi = med = 0.0
        else:
            lo = float(finite.quantile(0.01))
            hi = float(finite.quantile(0.99))
            if lo > hi:
                lo, hi = hi, lo
            med = float(finite.median())
        tr[col] = pd.to_numeric(tr[col], errors="coerce").clip(lo, hi).fillna(med)
        va[col] = pd.to_numeric(va[col], errors="coerce").clip(lo, hi).fillna(med)
        clip_meta[col] = {"p01": lo, "p99": hi, "median": med}
    return tr.astype(float), va.astype(float), clip_meta


def fit_model(model, X: pd.DataFrame, y: pd.Series):
    if isinstance(model, HistGradientBoostingClassifier):
        pos = max(int(y.sum()), 1)
        neg = max(int(len(y) - pos), 1)
        weights = np.where(np.asarray(y) == 1, neg / pos, 1.0)
        return model.fit(X, y, sample_weight=weights)
    return model.fit(X, y)


def metrics(y, score, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score >= threshold
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None,
        "alert_rate": float(pred.mean()),
        "green_npv": float(((y == 0) & (~pred)).sum() / max((~pred).sum(), 1)),
        "tp": int(((y == 1) & pred).sum()),
        "fp": int(((y == 0) & pred).sum()),
        "fn": int(((y == 1) & (~pred)).sum()),
        "tn": int(((y == 0) & (~pred)).sum()),
    }


def choose_threshold(y, score) -> tuple[float, dict, int]:
    candidates = np.unique(np.r_[np.linspace(0.05, 0.90, 171), np.quantile(score, np.linspace(0.03, 0.97, 95))])
    rows = []
    for t in candidates:
        m = metrics(y, score, float(t))
        penalty = 0.0
        penalty += 0.55 * max(0.75 - m["recall"], 0.0)
        penalty += 0.45 * max(m["alert_rate"] - 0.45, 0.0)
        penalty += 0.25 * max(0.93 - m["green_npv"], 0.0)
        objective = 0.40 * m["f1"] + 0.20 * m["balanced_accuracy"] + 0.20 * m["recall"] + 0.20 * m["precision"] - penalty
        rows.append((float(t), m, objective))
    feasible = [r for r in rows if r[1]["recall"] >= 0.75 and r[1]["alert_rate"] <= 0.45 and r[1]["green_npv"] >= 0.93]
    pool = feasible if feasible else rows
    pool.sort(key=lambda r: (r[2], r[1]["f1"], r[1]["balanced_accuracy"]), reverse=True)
    return pool[0][0], pool[0][1], len(feasible)


def outer_folds(meta: pd.DataFrame):
    windows = [
        ("2023-07-08", "2023-09-30"),
        ("2023-10-08", "2023-12-31"),
        ("2024-01-08", "2024-03-31"),
        ("2024-04-08", "2024-06-30"),
        ("2024-07-08", "2024-08-19"),
    ]
    out = []
    for fid, (start, end) in enumerate(windows):
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        train = meta.date <= start - pd.Timedelta(days=PURGE_DAYS + 1)
        val = meta.date.between(start, end)
        if train.sum() >= 100 and val.sum() >= 35 and meta.loc[train, "target"].nunique() == 2 and meta.loc[val, "target"].nunique() == 2:
            out.append((fid, start, end, train, val))
    return out


def nested_run(meta: pd.DataFrame, X: pd.DataFrame, feature_cols: list[str], scope: str) -> tuple[dict, pd.DataFrame]:
    fs = outer_folds(meta)
    if len(fs) != 5:
        raise RuntimeError(f"Expected 5 outer folds, got {len(fs)}")

    candidates = {name: [] for name in [*factories().keys(), "mean_ensemble"]}
    fold_info = []

    for fid, start, end, outer_train, outer_val in fs:
        outer_train_dates = meta.loc[outer_train, "date"]
        inner_end = outer_train_dates.max()
        inner_start = inner_end - pd.Timedelta(days=55)
        inner_train = outer_train & (meta.date <= inner_start - pd.Timedelta(days=PURGE_DAYS + 1))
        inner_val = outer_train & meta.date.between(inner_start, inner_end)
        if inner_train.sum() < 50 or inner_val.sum() < 20 or meta.loc[inner_train, "target"].nunique() < 2 or meta.loc[inner_val, "target"].nunique() < 2:
            raise RuntimeError(f"Fold {fid}: inner split invalid")

        Xin_tr, Xin_va, _ = fold_prepare(X.loc[inner_train, feature_cols], X.loc[inner_val, feature_cols])
        y_in_tr = meta.loc[inner_train, "target"].astype(int)
        y_in_va = meta.loc[inner_val, "target"].astype(int)
        inner_scores = {}
        inner_thresholds = {}
        inner_models = factories()
        for name, factory in inner_models.items():
            m = fit_model(clone(factory), Xin_tr, y_in_tr)
            score = m.predict_proba(Xin_va)[:, 1]
            t, tm, nf = choose_threshold(y_in_va, score)
            inner_scores[name] = score
            inner_thresholds[name] = {"threshold": t, "inner_metrics": tm, "feasible_thresholds": nf}
        ensemble_inner = np.column_stack([inner_scores[n] for n in factories()]).mean(axis=1)
        et, em, enf = choose_threshold(y_in_va, ensemble_inner)
        inner_thresholds["mean_ensemble"] = {"threshold": et, "inner_metrics": em, "feasible_thresholds": enf}

        Xout_tr, Xout_va, _ = fold_prepare(X.loc[outer_train, feature_cols], X.loc[outer_val, feature_cols])
        y_out_tr = meta.loc[outer_train, "target"].astype(int)
        y_out_va = meta.loc[outer_val, "target"].astype(int)
        outer_scores = {}
        for name, factory in factories().items():
            m = fit_model(clone(factory), Xout_tr, y_out_tr)
            outer_scores[name] = m.predict_proba(Xout_va)[:, 1]
        outer_scores["mean_ensemble"] = np.column_stack([outer_scores[n] for n in factories()]).mean(axis=1)

        for name, score in outer_scores.items():
            threshold = float(inner_thresholds[name]["threshold"])
            part = pd.DataFrame({
                "date": meta.loc[outer_val, "date"].to_numpy(),
                "target": y_out_va.to_numpy(),
                "score": score,
                "threshold": threshold,
                "fold_id": fid,
                "scope": scope,
                "model": name,
            })
            candidates[name].append(part)
        fold_info.append({
            "fold_id": fid,
            "start": str(start.date()),
            "end": str(end.date()),
            "outer_train_rows": int(outer_train.sum()),
            "outer_validation_rows": int(outer_val.sum()),
            "outer_positives": int(y_out_va.sum()),
            "inner_train_rows": int(inner_train.sum()),
            "inner_validation_rows": int(inner_val.sum()),
            "inner_start": str(inner_start.date()),
            "inner_end": str(inner_end.date()),
            "thresholds": inner_thresholds,
        })

    results = {}
    all_oof = []
    for name, parts in candidates.items():
        q = pd.concat(parts, ignore_index=True).sort_values(["fold_id", "date"]).reset_index(drop=True)
        pred = q.score.to_numpy(float) >= q.threshold.to_numpy(float)
        y = q.target.to_numpy(int)
        # Nested metrics use each fold's threshold chosen strictly from that fold's past inner validation.
        nested = {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, q.score)),
            "pr_auc": float(average_precision_score(y, q.score)),
            "alert_rate": float(pred.mean()),
            "green_npv": float(((y == 0) & (~pred)).sum() / max((~pred).sum(), 1)),
            "tp": int(((y == 1) & pred).sum()),
            "fp": int(((y == 0) & pred).sum()),
            "fn": int(((y == 1) & (~pred)).sum()),
            "tn": int(((y == 0) & (~pred)).sum()),
        }
        per_fold = []
        for fid, z in q.groupby("fold_id"):
            zz_pred = z.score.to_numpy(float) >= z.threshold.to_numpy(float)
            yy = z.target.to_numpy(int)
            per_fold.append({
                "fold_id": int(fid),
                "threshold": float(z.threshold.iloc[0]),
                "recall": float(recall_score(yy, zz_pred, zero_division=0)),
                "precision": float(precision_score(yy, zz_pred, zero_division=0)),
                "f1": float(f1_score(yy, zz_pred, zero_division=0)),
                "alert_rate": float(zz_pred.mean()),
                "green_npv": float(((yy == 0) & (~zz_pred)).sum() / max((~zz_pred).sum(), 1)),
                "roc_auc": float(roc_auc_score(yy, z.score)),
                "pr_auc": float(average_precision_score(yy, z.score)),
            })
        results[name] = {
            "nested_oof_metrics": nested,
            "per_fold": per_fold,
            "median_inner_threshold": float(q.groupby("fold_id").threshold.first().median()),
            "worst_fold_recall": float(min(x["recall"] for x in per_fold)),
            "max_fold_alert_rate": float(max(x["alert_rate"] for x in per_fold)),
        }
        all_oof.append(q)
    return {"scope": scope, "folds": fold_info, "models": results}, pd.concat(all_oof, ignore_index=True)


def selection_key(item: tuple[str, str, dict]) -> tuple:
    scope, model, result = item
    m = result["nested_oof_metrics"]
    feasible = m["recall"] >= 0.75 and m["alert_rate"] <= 0.45 and m["green_npv"] >= 0.93 and result["worst_fold_recall"] >= 0.50
    return (
        int(feasible),
        m["f1"],
        m["pr_auc"],
        m["balanced_accuracy"],
        m["roc_auc"],
        -m["alert_rate"],
    )


def fit_final(selected_scope: str, selected_model: str, meta: pd.DataFrame, X: pd.DataFrame, feature_cols: list[str], threshold: float):
    Xfit, _, prep = fold_prepare(X[feature_cols], X[feature_cols])
    y = meta.target.astype(int)
    if selected_model == "mean_ensemble":
        models = {name: fit_model(clone(factory), Xfit, y) for name, factory in factories().items()}
        fitted = {"kind": "mean_ensemble", "models": models}
    else:
        fitted = fit_model(clone(factories()[selected_model]), Xfit, y)
    artifact = {
        "version": VERSION,
        "status": "DEVELOPMENT_NESTED_ROLLING_ORIGIN_EXTERNAL_VALIDATION_REQUIRED",
        "scope": selected_scope,
        "model": selected_model,
        "threshold": float(threshold),
        "decline_definition": "next_7_day_sales < 85% of trailing_28_day daily baseline x 7",
        "feature_columns": feature_cols,
        "preprocessing": prep,
        "model_object": fitted,
    }
    joblib.dump(artifact, MODEL)


def main():
    meta, X, merchant_cols, full_cols = build_panel()
    if len(meta) < 500:
        raise RuntimeError(f"Unexpected merchant rows: {len(meta)}")
    panel_out = pd.concat([meta, X], axis=1)
    panel_out.to_csv(PANEL, index=False)

    merchant_run, merchant_oof = nested_run(meta, X, merchant_cols, "merchant_only")
    full_run, full_oof = nested_run(meta, X, full_cols, "merchant_plus_category_signals")
    oof = pd.concat([merchant_oof, full_oof], ignore_index=True)
    oof.to_csv(OOF, index=False)

    choices = []
    for run in (merchant_run, full_run):
        for model, result in run["models"].items():
            choices.append((run["scope"], model, result))
    selected_scope, selected_model, selected_result = max(choices, key=selection_key)
    selected_cols = merchant_cols if selected_scope == "merchant_only" else full_cols
    deployment_threshold = selected_result["median_inner_threshold"]

    v61 = None
    if V61_REPORT.exists():
        try:
            v61_raw = json.loads(V61_REPORT.read_text(encoding="utf-8"))
            v61 = v61_raw.get("selected_policy", {}).get("metrics")
        except Exception:
            v61 = None

    full_best = max(full_run["models"].items(), key=lambda z: selection_key(("merchant_plus_category_signals", z[0], z[1])))
    merchant_best = max(merchant_run["models"].items(), key=lambda z: selection_key(("merchant_only", z[0], z[1])))
    full_auc = full_best[1]["nested_oof_metrics"]["roc_auc"]
    merchant_auc = merchant_best[1]["nested_oof_metrics"]["roc_auc"]
    category_delta_auc = full_auc - merchant_auc

    selected_metrics = selected_result["nested_oof_metrics"]
    gates = {
        "five_outer_rolling_origin_folds": len(full_run["folds"]) == 5,
        "inner_thresholds_use_past_only": True,
        "seven_day_target_purge": True,
        "no_blind_holdout_reuse_claim": True,
        "no_smote_or_synthetic_oversampling": True,
        "recall_ge_075": selected_metrics["recall"] >= 0.75,
        "green_npv_ge_093": selected_metrics["green_npv"] >= 0.93,
        "alert_rate_le_045": selected_metrics["alert_rate"] <= 0.45,
        "worst_fold_recall_ge_050": selected_result["worst_fold_recall"] >= 0.50,
        "category_signal_ablation_nonnegative_auc": category_delta_auc >= 0.0,
    }

    fit_final(selected_scope, selected_model, meta, X, selected_cols, deployment_threshold)

    manifest = {
        "merchant_feature_count": len(merchant_cols),
        "category_signal_feature_count": len([c for c in full_cols if c not in merchant_cols]),
        "full_feature_count": len(full_cols),
        "merchant_features": merchant_cols,
        "category_signal_features": [c for c in full_cols if c not in merchant_cols],
    }
    FEATURE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "version": VERSION,
        "scientific_status": "DEVELOPMENT_NESTED_ROLLING_ORIGIN_EXTERNAL_SAUDI_VALIDATION_REQUIRED",
        "merchant_rows": len(meta),
        "early_positive_rate": float(meta.target.mean()),
        "early_decline_threshold": DECLINE,
        "purge_days": PURGE_DAYS,
        "method": "Category/sector panel is used only to derive past-available daily merchant-level signals; final target remains merchant-total next-7-day decline.",
        "merchant_only": merchant_run,
        "merchant_plus_category_signals": full_run,
        "ablation": {
            "best_merchant_only_model": merchant_best[0],
            "best_full_model": full_best[0],
            "merchant_only_roc_auc": merchant_auc,
            "merchant_plus_category_roc_auc": full_auc,
            "category_signal_delta_roc_auc": category_delta_auc,
        },
        "selected": {
            "scope": selected_scope,
            "model": selected_model,
            "deployment_threshold_from_median_inner_thresholds": deployment_threshold,
            **selected_result,
        },
        "v6_1_reference_metrics": v61,
        "development_gates": gates,
        "all_development_gates_passed": bool(all(gates.values())),
        "limitations": [
            "This is nested rolling-origin development evidence, not a new independent blind holdout.",
            "The V7.0 blind window was already opened and is not reused as independent evidence.",
            "Transaction microdata remain Saudi-localized rather than directly observed Saudi merchant transactions.",
            "External real Saudi merchant longitudinal validation is still required.",
        ],
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    m = selected_metrics
    lines = [
        "# Sales Sentinel V7.1 — Merchant + Category Signals (Nested Rolling-Origin)",
        "",
        f"- Merchant rows: **{len(meta):,}**",
        f"- Positive rate: **{meta.target.mean():.2%}**",
        f"- Merchant-only features: **{len(merchant_cols)}**",
        f"- Added category-regime signals: **{len(full_cols) - len(merchant_cols)}**",
        f"- Selected scope: **{selected_scope}**",
        f"- Selected model: **{selected_model}**",
        f"- Deployment threshold (median inner thresholds): **{deployment_threshold:.3f}**",
        f"- Nested OOF ROC-AUC: **{m['roc_auc']:.2%}**",
        f"- Nested OOF PR-AUC: **{m['pr_auc']:.2%}**",
        f"- Balanced Accuracy: **{m['balanced_accuracy']:.2%}**",
        f"- Precision: **{m['precision']:.2%}**",
        f"- Recall: **{m['recall']:.2%}**",
        f"- F1: **{m['f1']:.2%}**",
        f"- GREEN NPV: **{m['green_npv']:.2%}**",
        f"- Alert rate: **{m['alert_rate']:.2%}**",
        f"- TP / FP / FN / TN: **{m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}**",
        f"- Worst-fold recall: **{selected_result['worst_fold_recall']:.2%}**",
        f"- Max-fold alert rate: **{selected_result['max_fold_alert_rate']:.2%}**",
        f"- Category-signal AUC delta vs merchant-only: **{category_delta_auc:+.2%}**",
        f"- Development gates passed: **{all(gates.values())}**",
        "- Independent real Saudi merchant validation: **Pending**",
        "",
        "Scientific boundary: V7.1 uses nested rolling-origin development evaluation. The previously opened V7.0 blind window is not reused or described as independent evidence.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
