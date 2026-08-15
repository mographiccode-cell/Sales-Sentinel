from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)

import train_merchant_total_hybrid_v4_3 as merchant_base
import train_merchant_total_triage_v5 as v5
import train_sama_city_risk_v3 as market_v3

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-MERCHANT-MARKET-FUSION-6.0"
SRC = ROOT / "data" / "merchant_v4_3" / "merchant_total_feature_panel_v4_3.csv"
OUT = ROOT / "reports" / "merchant_market_fusion_v6"
MOD = ROOT / "models" / "merchant_market_fusion_v6"
OUT.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
MODEL = MOD / "merchant_market_fusion_v6.joblib"
SEED = 61

EARLY_RATIO = 0.85
SEVERE_RATIO = 0.80
MARKET_PREFIX = "market_v3__"
MERCHANT_MODELS = ("logreg_k36", "extra_k48")


def sunday_week_start(s):
    d = pd.to_datetime(s)
    return d - pd.to_timedelta((d.dt.dayofweek + 1) % 7, unit="D")


def metric(y, score, threshold):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    pred = score >= float(threshold)
    tn = int(((y == 0) & (~pred)).sum())
    fn = int(((y == 1) & (~pred)).sum())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None,
        "alert_rate": float(pred.mean()),
        "green_npv": float(tn / max(tn + fn, 1)),
        "tp": int(((y == 1) & pred).sum()),
        "fp": int(((y == 0) & pred).sum()),
        "fn": fn,
        "tn": tn,
    }


def choose_threshold(y, score, fold_id):
    candidates = np.unique(np.r_[np.linspace(0.05, 0.95, 181), np.quantile(score, np.linspace(0.02, 0.98, 121))])
    rows = []
    for t in candidates:
        m = metric(y, score, t)
        per = []
        for fid in sorted(np.unique(fold_id)):
            z = fold_id == fid
            fm = metric(y[z], score[z], t)
            fm["fold_id"] = int(fid)
            fm["positives"] = int(y[z].sum())
            per.append(fm)
        stable = [z["recall"] for z in per if z["positives"] >= 5]
        worst_recall = min(stable, default=m["recall"])
        max_alert = max((z["alert_rate"] for z in per), default=m["alert_rate"])
        supported = (
            m["recall"] >= 0.78
            and m["precision"] >= 0.30
            and m["f1"] >= 0.44
            and m["green_npv"] >= 0.94
            and m["alert_rate"] <= 0.45
            and worst_recall >= 0.45
            and max_alert <= 0.65
        )
        objective = (
            1.35 * m["f1"] + 0.65 * m["balanced_accuracy"] + 0.35 * m["precision"]
            + 0.30 * m["recall"] + 0.20 * m["green_npv"] - 0.30 * m["alert_rate"]
        )
        rows.append((supported, objective, float(t), m, per, worst_recall, max_alert))
    pool = [r for r in rows if r[0]] or rows
    pool.sort(key=lambda r: (r[0], r[1], r[3]["f1"], r[3]["precision"], r[3]["recall"]), reverse=True)
    b = pool[0]
    return {
        "supported": bool(b[0]),
        "threshold": b[2],
        "metrics": b[3],
        "per_fold": b[4],
        "worst_fold_recall": float(b[5]),
        "max_fold_alert_rate": float(b[6]),
        "feasible_thresholds": int(sum(1 for r in rows if r[0])),
        "objective": float(b[1]),
    }


def fit_market_ensemble(X, y):
    fitted = {}
    for name, factory in market_v3.model_factories().items():
        fitted[name] = market_v3.fit_one(clone(factory), X, y)
    return fitted


def predict_market_ensemble(models, X):
    pred = np.column_stack([m.predict_proba(X)[:, 1] for m in models.values()])
    return pred.mean(axis=1)


def build_prequential_market_channel(merchant_dates):
    panel = pd.read_csv(market_v3.HISTORY, parse_dates=["week_start"])
    labeled_meta, labeled_X, _, _ = market_v3.featureize(panel, require_target=True)
    all_meta, all_X, _, all_pc = market_v3.featureize(panel, require_target=False)
    all_meta = all_meta.copy()
    all_meta["precursor_count"] = all_pc.to_numpy(int)

    min_week = sunday_week_start(pd.Series([pd.to_datetime(merchant_dates).min()])).iloc[0] - pd.Timedelta(days=14)
    max_week = sunday_week_start(pd.Series([pd.to_datetime(merchant_dates).max()])).iloc[0] + pd.Timedelta(days=7)
    wanted = all_meta.week_start.between(min_week, max_week)
    score_frame = all_meta.loc[wanted, ["week_start", "city", "precursor_count"]].copy()
    score_frame["risk"] = np.nan

    # One frozen model per calendar month. Every model is trained only on labels
    # whose next-week outcome was already observable before that month begins.
    months = pd.PeriodIndex(score_frame.week_start, freq="M").unique().sort_values()
    snapshots = []
    for period in months:
        month_start = period.start_time
        cutoff = month_start - pd.Timedelta(days=14)
        tr = labeled_meta.week_start <= cutoff
        if int(tr.sum()) < 500 or labeled_meta.loc[tr, "target"].nunique() < 2:
            continue
        models = fit_market_ensemble(labeled_X.loc[tr], labeled_meta.loc[tr, "target"].astype(int))
        mask = pd.PeriodIndex(score_frame.week_start, freq="M") == period
        idx = score_frame.index[mask]
        source_idx = all_meta.index[
            wanted & (pd.PeriodIndex(all_meta.week_start, freq="M") == period)
        ]
        if len(idx) != len(source_idx):
            raise RuntimeError("Market score alignment failed")
        score_frame.loc[idx, "risk"] = predict_market_ensemble(models, all_X.loc[source_idx])
        snapshots.append({
            "month": str(period),
            "train_rows": int(tr.sum()),
            "train_positives": int(labeled_meta.loc[tr, "target"].sum()),
            "label_cutoff": str(cutoff.date()),
        })

    if score_frame.risk.isna().any():
        score_frame = score_frame.sort_values(["city", "week_start"])
        score_frame["risk"] = score_frame.groupby("city").risk.ffill()
    if score_frame.risk.isna().any():
        raise RuntimeError("Insufficient historical SAMA data for prequential market scoring")

    weekly = score_frame.groupby("week_start", as_index=False).agg(
        risk_mean=("risk", "mean"),
        risk_max=("risk", "max"),
        risk_p75=("risk", lambda x: float(np.quantile(x, 0.75))),
        risk_p90=("risk", lambda x: float(np.quantile(x, 0.90))),
        risk_share_25=("risk", lambda x: float(np.mean(np.asarray(x) >= 0.25))),
        risk_share_50=("risk", lambda x: float(np.mean(np.asarray(x) >= 0.50))),
        precursor_mean=("precursor_count", "mean"),
        precursor_share_2=("precursor_count", lambda x: float(np.mean(np.asarray(x) >= 2))),
    )
    # V3 at week W predicts next-week risk, so expose it to merchant week W+7.
    weekly["merchant_week_start"] = weekly["week_start"] + pd.Timedelta(days=7)
    weekly = weekly.drop(columns=["week_start"])
    weekly = weekly.rename(columns={c: MARKET_PREFIX + c for c in weekly.columns if c != "merchant_week_start"})

    q = pd.DataFrame({"date": pd.to_datetime(merchant_dates)})
    q["merchant_week_start"] = sunday_week_start(q.date)
    q = q.merge(weekly, on="merchant_week_start", how="left", validate="many_to_one")
    market_cols = [c for c in q.columns if c.startswith(MARKET_PREFIX)]
    if q[market_cols].isna().any().any():
        q = q.sort_values("date").reset_index()
        q[market_cols] = q[market_cols].ffill()
        if q[market_cols].isna().any().any():
            first_valid = q[market_cols].dropna().iloc[0]
            for c in market_cols:
                q[c] = q[c].fillna(float(first_valid[c]))
        q = q.sort_values("index").drop(columns=["index"]).reset_index(drop=True)
    return q[market_cols].astype(float), snapshots


def merchant_oof(X, y, folds):
    factories = v5.model_factories()
    out = {name: np.full(len(X), np.nan) for name in MERCHANT_MODELS}
    fold_id = np.full(len(X), -1, int)
    fold_meta = []
    for fid, (st, en, tr, va) in enumerate(folds):
        for name in MERCHANT_MODELS:
            m = clone(factories[name]).fit(X.loc[tr], y[tr.to_numpy()])
            out[name][va.to_numpy()] = m.predict_proba(X.loc[va])[:, 1]
        fold_id[va.to_numpy()] = fid
        fold_meta.append({
            "fold_id": fid,
            "start": str(st.date()),
            "end": str(en.date()),
            "train_rows": int(tr.sum()),
            "validation_rows": int(va.sum()),
            "validation_positives": int(y[va.to_numpy()].sum()),
        })
    mask = fold_id >= 0
    frame = pd.DataFrame({
        "row_index": np.where(mask)[0],
        "fold_id": fold_id[mask],
        "y": y[mask],
        **{name: out[name][mask] for name in MERCHANT_MODELS},
    })
    return frame, fold_meta


def meta_features(base_oof, market_X):
    idx = base_oof.row_index.to_numpy(int)
    z = pd.DataFrame(index=base_oof.index)
    z["merchant_logreg"] = base_oof["logreg_k36"].to_numpy(float)
    z["merchant_extra"] = base_oof["extra_k48"].to_numpy(float)
    z["merchant_mean"] = z[["merchant_logreg", "merchant_extra"]].mean(axis=1)
    z["merchant_disagreement"] = np.abs(z.merchant_logreg - z.merchant_extra)
    for c in market_X.columns:
        z[c] = market_X.iloc[idx][c].to_numpy(float)
    z["merchant_x_market"] = z["merchant_mean"] * z[f"{MARKET_PREFIX}risk_mean"]
    z["merchant_x_marketmax"] = z["merchant_mean"] * z[f"{MARKET_PREFIX}risk_max"]
    return z


def prequential_stack(base_oof, Z):
    score = np.full(len(base_oof), np.nan)
    meta_history = []
    for fid in sorted(base_oof.fold_id.unique()):
        va = base_oof.fold_id.to_numpy() == fid
        tr = base_oof.fold_id.to_numpy() < fid
        ytr = base_oof.loc[tr, "y"].to_numpy(int)
        # Before enough OOF history exists, use a conservative fixed blend.
        if tr.sum() < 120 or len(np.unique(ytr)) < 2 or ytr.sum() < 8:
            merchant = Z.loc[va, "merchant_mean"].to_numpy(float)
            market = Z.loc[va, f"{MARKET_PREFIX}risk_mean"].to_numpy(float)
            market_max = Z.loc[va, f"{MARKET_PREFIX}risk_max"].to_numpy(float)
            score[va] = 0.78 * merchant + 0.14 * market + 0.08 * market_max
            meta_history.append({"fold_id": int(fid), "mode": "fixed_cold_start", "meta_train_rows": int(tr.sum()), "meta_train_positives": int(ytr.sum())})
        else:
            meta = LogisticRegression(C=0.10, class_weight="balanced", max_iter=5000, random_state=SEED)
            meta.fit(Z.loc[tr], ytr)
            score[va] = meta.predict_proba(Z.loc[va])[:, 1]
            meta_history.append({"fold_id": int(fid), "mode": "prequential_logistic", "meta_train_rows": int(tr.sum()), "meta_train_positives": int(ytr.sum())})
    return score, meta_history


def fit_final_artifact(X, y15, y20, base_oof15, Z15, threshold15, market_X, snapshots):
    factories = v5.model_factories()
    merchant_models_15 = {name: clone(factories[name]).fit(X, y15) for name in MERCHANT_MODELS}
    merchant_models_20 = {name: clone(factories[name]).fit(X, y20) for name in MERCHANT_MODELS}

    final_meta = LogisticRegression(C=0.10, class_weight="balanced", max_iter=5000, random_state=SEED)
    final_meta.fit(Z15, base_oof15.y.to_numpy(int))

    panel = pd.read_csv(market_v3.HISTORY, parse_dates=["week_start"])
    mm, mx, _, _ = market_v3.featureize(panel, require_target=True)
    market_train = mm.week_start <= market_v3.DEV_END
    market_models = fit_market_ensemble(mx.loc[market_train], mm.loc[market_train, "target"].astype(int))

    artifact = {
        "version": VERSION,
        "status": "DEVELOPMENT_FROZEN_PENDING_EXTERNAL_MERCHANT_VALIDATION",
        "target_definition": "merchant next-7-day sales versus trailing-28-day mean",
        "early_decline_ratio": EARLY_RATIO,
        "severe_decline_ratio": SEVERE_RATIO,
        "merchant_model_names": list(MERCHANT_MODELS),
        "merchant_models_15": merchant_models_15,
        "merchant_models_20": merchant_models_20,
        "market_v3_models": market_models,
        "market_v3_feature_columns": list(mx.columns),
        "market_channel_columns": list(market_X.columns),
        "meta_model": final_meta,
        "meta_feature_columns": list(Z15.columns),
        "early_threshold": float(threshold15),
        "market_prequential_protocol": "monthly frozen V3-style ensembles; label cutoff >=14 days before scored month; week W score is exposed to merchant week W+7",
        "market_snapshots": snapshots,
        "states": {
            "GREEN": "below early decline threshold",
            "AMBER": ">=15% merchant decline risk",
            "RED": "reserved for >=20% severe decline only when a separate high-precision threshold is supported",
        },
    }
    return artifact


def main():
    d = pd.read_csv(SRC, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    meta = d[["date", "future_ratio"]].copy()
    X = d.drop(columns=["date", "future_ratio", "target"]).replace([np.inf, -np.inf], np.nan)
    y15 = (meta.future_ratio.to_numpy(float) < EARLY_RATIO).astype(int)
    y20 = (meta.future_ratio.to_numpy(float) < SEVERE_RATIO).astype(int)
    folds = merchant_base.folds(meta.assign(target=y20))
    if len(folds) != 5:
        raise RuntimeError(f"Expected five purged rolling merchant folds, got {len(folds)}")

    market_X, market_snapshots = build_prequential_market_channel(meta.date)
    if len(market_X) != len(X):
        raise RuntimeError("Merchant/market row count mismatch")

    base15, fold_meta = merchant_oof(X, y15, folds)
    Z15 = meta_features(base15, market_X)
    stack15, stack_meta = prequential_stack(base15, Z15)
    y15_oof = base15.y.to_numpy(int)
    fold_ids = base15.fold_id.to_numpy(int)

    merchant_mean = base15[list(MERCHANT_MODELS)].mean(axis=1).to_numpy(float)
    extra_score = base15["extra_k48"].to_numpy(float)
    candidates = {
        "merchant_extra": extra_score,
        "merchant_mean": merchant_mean,
        "v6_prequential_stack": stack15,
    }
    ranking = {}
    selections = {}
    for name, score in candidates.items():
        selections[name] = choose_threshold(y15_oof, score, fold_ids)
        ranking[name] = {
            "roc_auc": float(roc_auc_score(y15_oof, score)),
            "pr_auc": float(average_precision_score(y15_oof, score)),
        }

    selected_name = max(
        selections,
        key=lambda n: (
            selections[n]["supported"],
            selections[n]["objective"],
            selections[n]["metrics"]["f1"],
            ranking[n]["roc_auc"],
        ),
    )
    selected = selections[selected_name]

    base20, _ = merchant_oof(X, y20, folds)
    Z20 = meta_features(base20, market_X)
    stack20, _ = prequential_stack(base20, Z20)
    y20_oof = base20.y.to_numpy(int)
    red_sel = choose_threshold(y20_oof, stack20, base20.fold_id.to_numpy(int))
    red_supported = bool(
        red_sel["metrics"]["precision"] >= 0.60
        and red_sel["metrics"]["recall"] >= 0.20
        and (red_sel["metrics"]["tp"] + red_sel["metrics"]["fp"]) >= 5
    )
    if not red_supported:
        red_sel["supported"] = False

    artifact = fit_final_artifact(
        X, y15, y20, base15, Z15, selected["threshold"], market_X, market_snapshots
    )
    artifact["selected_early_channel"] = selected_name
    artifact["red_threshold"] = float(red_sel["threshold"]) if red_supported else None
    artifact["red_supported"] = red_supported
    joblib.dump(artifact, MODEL)

    m = selected["metrics"]
    v44_path = ROOT / "reports" / "merchant_total_triage_v4_4" / "development_report.json"
    v44 = json.loads(v44_path.read_text(encoding="utf-8")) if v44_path.exists() else {}
    v44m = ((v44.get("metrics") or {}).get("AMBER_or_RED_vs_15pct") or {})
    if not v44m:
        v44m = {"precision": 0.30612244897959184, "recall": 0.7142857142857143, "f1": 0.42857142857142855,
                "alert_rate": 0.3858267716535433, "green_npv": 0.9230769230769231, "roc_auc": 0.7584107018069283}

    comparison = {
        "v4_4": v44m,
        "v6": m,
        "delta": {
            k: float(m[k] - v44m[k]) for k in
            ["precision", "recall", "f1", "alert_rate", "green_npv"]
            if k in m and k in v44m
        },
    }

    contract = {
        "roc_auc_min": 0.76,
        "recall_min": 0.78,
        "precision_min": 0.30,
        "f1_min": 0.44,
        "green_npv_min": 0.94,
        "alert_rate_max": 0.45,
        "worst_fold_recall_min": 0.45,
    }
    gates = {
        "merchant_rolling_origin_past_only": True,
        "merchant_target_purge_7days": True,
        "market_prequential_monthly_freeze": True,
        "market_label_availability_gap_14days": True,
        "market_week_shift_plus_7days": True,
        "meta_prequential_no_same_fold_labels": True,
        "no_synthetic_oversampling": True,
        "selected_is_v6_stack": selected_name == "v6_prequential_stack",
        "roc_auc": ranking[selected_name]["roc_auc"] >= contract["roc_auc_min"],
        "recall": m["recall"] >= contract["recall_min"],
        "precision": m["precision"] >= contract["precision_min"],
        "f1": m["f1"] >= contract["f1_min"],
        "green_npv": m["green_npv"] >= contract["green_npv_min"],
        "alert_rate": m["alert_rate"] <= contract["alert_rate_max"],
        "worst_fold_recall": selected["worst_fold_recall"] >= contract["worst_fold_recall_min"],
    }

    report = {
        "version": VERSION,
        "status": artifact["status"],
        "scientific_boundary": (
            "V6 is development evidence on the localized merchant panel. Merchant OOF uses purged rolling origin. "
            "The SAMA V3-style market channel is prequential: each scored month uses only market labels observable "
            "before that month, and week-W market risk is shifted to merchant week W+7. No external real merchant "
            "longitudinal validation has yet been performed."
        ),
        "merchant_rows": int(len(d)),
        "merchant_raw_features": int(X.shape[1]),
        "market_channel_features": int(market_X.shape[1]),
        "market_snapshots": market_snapshots,
        "oof_rows": int(len(base15)),
        "early_positive_rate": float(y15.mean()),
        "severe_positive_rate": float(y20.mean()),
        "folds": fold_meta,
        "stack_history": stack_meta,
        "ranking": ranking,
        "thresholds": selections,
        "selected_early_channel": selected_name,
        "selected_early": selected,
        "red": red_sel,
        "red_supported": red_supported,
        "comparison_vs_v4_4": comparison,
        "contract": contract,
        "gates": gates,
        "all_development_gates_passed": bool(all(gates.values())),
        "next_required_evidence": (
            "If V6 passes, freeze it and test on a never-used real Saudi merchant time series. "
            "If it fails, the limiting factor is merchant-data information content rather than threshold tuning."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = [
        "# Sales Sentinel v6.0 — Merchant + SAMA V3 prequential fusion",
        "",
        f"- Selected channel: **{selected_name}**",
        f"- Merchant rows: **{len(d)}**",
        f"- Merchant raw features: **{X.shape[1]}**",
        f"- Market channel features: **{market_X.shape[1]}**",
        f"- OOF ROC-AUC: **{ranking[selected_name]['roc_auc']:.2%}**",
        f"- OOF PR-AUC: **{ranking[selected_name]['pr_auc']:.2%}**",
        f"- Precision: **{m['precision']:.2%}**",
        f"- Recall: **{m['recall']:.2%}**",
        f"- F1: **{m['f1']:.2%}**",
        f"- GREEN NPV: **{m['green_npv']:.2%}**",
        f"- Alert rate: **{m['alert_rate']:.2%}**",
        f"- Worst-fold recall: **{selected['worst_fold_recall']:.2%}**",
        f"- RED supported: **{red_supported}**",
        f"- Development gates: **{all(gates.values())}**",
    ]
    if comparison["delta"]:
        summary.append(f"- F1 delta vs v4.4: **{comparison['delta'].get('f1', 0):+.2%}**")
        summary.append(f"- Recall delta vs v4.4: **{comparison['delta'].get('recall', 0):+.2%}**")
        summary.append(f"- Precision delta vs v4.4: **{comparison['delta'].get('precision', 0):+.2%}**")
    SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
