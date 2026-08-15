from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
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
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V7.4-COMPACT-PERCENTILE-CALIBRATED"
SEED = 42
PURGE_DAYS = 7
PANEL = ROOT / "data" / "merchant_v7_1" / "merchant_feature_panel_v7_1.csv"
V61_REPORT = ROOT / "reports" / "merchant_market_fusion_v6_1" / "development_report.json"
OUT = ROOT / "reports" / "merchant_compact_v7_4"
MOD = ROOT / "models" / "merchant_compact_v7_4"
OUT.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_predictions.csv"
MODEL = MOD / "merchant_compact_v7_4.joblib"


def uniq(xs):
    seen = set(); out = []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def feature_sets(columns: list[str]) -> dict[str, list[str]]:
    c = set(columns)
    exact_core = [
        "merchant__net_sales_sar__ratio_mean_7",
        "merchant__net_sales_sar__ratio_mean_14",
        "merchant__net_sales_sar__ratio_mean_28",
        "merchant__net_sales_sar__z_7",
        "merchant__net_sales_sar__z_28",
        "merchant__net_sales_sar__change_1",
        "merchant__net_sales_sar__change_7",
        "merchant__net_sales_sar__change_14",
        "merchant__net_sales_sar__change_28",
        "merchant__sales__ma7_vs_ma28",
        "merchant__sales__ma14_vs_ma56",
        "merchant__sales__vol7_vs_vol28",
        "merchant__sales__drawdown28",
        "merchant__invoice_count__ratio_mean_7",
        "merchant__invoice_count__ratio_mean_28",
        "merchant__invoice_count__change_7",
        "merchant__invoice_count__change_28",
        "merchant__unique_observed_customers__ratio_mean_7",
        "merchant__unique_observed_customers__ratio_mean_28",
        "merchant__unique_observed_customers__change_7",
        "merchant__average_invoice_value_sar__ratio_mean_7",
        "merchant__average_invoice_value_sar__change_7",
        "merchant__return_rate_value__change_7",
        "market__sama_weekly_market_index__change_7",
        "calendar__salary_period",
        "calendar__year_sin",
        "calendar__year_cos",
    ]
    regime = [
        "catregime__share_sales_below_090_ma7",
        "catregime__share_sales_below_100_ma7",
        "catregime__share_sales_above_110_ma7",
        "catregime__share_negative_sales_change7",
        "catregime__share_severe_sales_change7",
        "catregime__share_negative_invoice_change7",
        "catregime__share_negative_customer_change7",
        "catregime__sales_share_hhi",
        "catregime__largest_category_share",
        "catregime__share_sales_below_090_ma7__delta7",
        "catregime__share_negative_sales_change7__delta7",
        "catregime__share_negative_invoice_change7__delta7",
        "catregime__share_negative_customer_change7__delta7",
    ]
    agg = [
        "catagg__net_sales_ratio_mean_7__mean",
        "catagg__net_sales_ratio_mean_7__p10",
        "catagg__net_sales_ratio_mean_28__mean",
        "catagg__net_sales_change_7__mean",
        "catagg__net_sales_change_7__p10",
        "catagg__invoice_count_change_7__mean",
        "catagg__invoice_count_change_7__p10",
        "catagg__observed_customer_count_change_7__mean",
        "catagg__observed_customer_count_change_7__p10",
        "catagg__avg_invoice_value_change_7__mean",
        "catagg__return_rate_value_change_7__mean",
        "catagg__category_share_change_7__std",
        "catagg__sama_predicted_value_h1_change_vs_last__mean",
        "catagg__sama_predicted_value_h2_change_vs_last__mean",
        "catagg__sama_predicted_count_h1_change_vs_last__mean",
    ]
    core = [x for x in exact_core if x in c]
    reg = [x for x in regime if x in c]
    ag = [x for x in agg if x in c]
    if len(core) < 20:
        raise RuntimeError(f"Compact core unexpectedly small: {len(core)}")
    sets = {
        "compact24": core[:24],
        "compact32": uniq(core[:24] + reg[:8]),
        "compact40": uniq(core[:25] + reg[:9] + ag[:6]),
        "compact48": uniq(core + reg[:11] + ag[:10]),
    }
    return sets


def prepare(Xtr: pd.DataFrame, Xva: pd.DataFrame):
    tr = Xtr.copy(); va = Xva.copy(); prep = {}
    for col in tr.columns:
        a = pd.to_numeric(tr[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = a.dropna()
        if finite.empty:
            lo = hi = med = 0.0
        else:
            lo = float(finite.quantile(.01)); hi = float(finite.quantile(.99)); med = float(finite.median())
        tr[col] = pd.to_numeric(tr[col], errors="coerce").clip(lo, hi).fillna(med)
        va[col] = pd.to_numeric(va[col], errors="coerce").clip(lo, hi).fillna(med)
        prep[col] = {"p01": lo, "p99": hi, "median": med}
    return tr.astype(float), va.astype(float), prep


def make_model(name: str, y: pd.Series):
    pos = max(int(y.sum()), 1); neg = max(int(len(y) - pos), 1); spw = neg / pos
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=.08, class_weight="balanced", max_iter=5000, random_state=SEED))
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=900, max_depth=5, min_samples_leaf=8, max_features=.65, class_weight="balanced", random_state=SEED, n_jobs=-1)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=320, learning_rate=.025, max_leaf_nodes=10, min_samples_leaf=20, l2_regularization=15.0, random_state=SEED)
    if name == "xgb":
        return XGBClassifier(n_estimators=450, max_depth=2, learning_rate=.025, min_child_weight=8, subsample=.85, colsample_bytree=.75, reg_alpha=1.5, reg_lambda=12.0, gamma=.15, objective="binary:logistic", eval_metric="logloss", scale_pos_weight=spw, random_state=SEED, n_jobs=2)
    if name == "catboost":
        return CatBoostClassifier(iterations=450, depth=4, learning_rate=.025, l2_leaf_reg=12.0, random_seed=SEED, auto_class_weights="Balanced", verbose=False, allow_writing_files=False)
    raise KeyError(name)


def fit_model(name: str, X: pd.DataFrame, y: pd.Series):
    m = make_model(name, y)
    if name == "hist_gb":
        pos = max(int(y.sum()), 1); neg = max(int(len(y) - pos), 1)
        w = np.where(np.asarray(y) == 1, neg / pos, 1.0)
        return m.fit(X, y, sample_weight=w)
    return m.fit(X, y)


def percentile_score(score, reference):
    ref = np.sort(np.asarray(reference, float))
    return np.searchsorted(ref, np.asarray(score, float), side="right") / max(len(ref), 1)


def metric(y, pred, rank_score=None):
    y = np.asarray(y, int); pred = np.asarray(pred, bool)
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & (~pred)).sum()); tn = int(((y == 0) & (~pred)).sum())
    d = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alert_rate": float(pred.mean()),
        "green_npv": float(tn / max(tn + fn, 1)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if rank_score is not None and len(np.unique(y)) == 2:
        d["roc_auc"] = float(roc_auc_score(y, rank_score)); d["pr_auc"] = float(average_precision_score(y, rank_score))
    return d


def choose_q(y, rank):
    rows = []
    for q in np.arange(.52, .901, .01):
        m = metric(y, np.asarray(rank) >= q, rank)
        feasible = m["recall"] >= .75 and m["green_npv"] >= .93 and m["alert_rate"] <= .45
        objective = .42*m["f1"] + .18*m["precision"] + .18*m["recall"] + .12*m["balanced_accuracy"] + .10*m["green_npv"] - .25*max(m["alert_rate"]-.40, 0)
        rows.append((feasible, objective, float(q), m))
    feasible_rows = [r for r in rows if r[0]]
    pool = feasible_rows or rows
    pool.sort(key=lambda r: (r[1], r[3]["f1"], r[3]["precision"], -r[3]["alert_rate"]), reverse=True)
    return pool[0][2], pool[0][3], len(feasible_rows)


def folds(d):
    windows = [
        ("2023-07-08", "2023-09-30"),
        ("2023-10-08", "2023-12-31"),
        ("2024-01-08", "2024-03-31"),
        ("2024-04-08", "2024-06-30"),
        ("2024-07-08", "2024-08-19"),
    ]
    out = []
    for fid, (a,b) in enumerate(windows):
        a = pd.Timestamp(a); b = pd.Timestamp(b)
        tr = d.date <= a - pd.Timedelta(days=PURGE_DAYS+1)
        va = d.date.between(a,b)
        out.append((fid,a,b,tr,va))
    return out


def evaluate_candidate(d, features, model_name, set_name):
    parts = []; fold_details = []
    for fid, a, b, otr, ova in folds(d):
        train_dates = d.loc[otr, "date"]
        inner_end = train_dates.max(); inner_start = inner_end - pd.Timedelta(days=55)
        itr = otr & (d.date <= inner_start - pd.Timedelta(days=PURGE_DAYS+1))
        iva = otr & d.date.between(inner_start, inner_end)
        if itr.sum() < 50 or iva.sum() < 20:
            raise RuntimeError(f"Invalid inner split fold {fid}")
        Xit, Xiv, _ = prepare(d.loc[itr, features], d.loc[iva, features])
        yit = d.loc[itr, "target"].astype(int); yiv = d.loc[iva, "target"].astype(int)
        im = fit_model(model_name, Xit, yit)
        inner_train_score = im.predict_proba(Xit)[:,1]
        inner_val_score = im.predict_proba(Xiv)[:,1]
        inner_rank = percentile_score(inner_val_score, inner_train_score)
        q, qm, nfeas = choose_q(yiv, inner_rank)

        Xot, Xov, _ = prepare(d.loc[otr, features], d.loc[ova, features])
        yot = d.loc[otr, "target"].astype(int); yov = d.loc[ova, "target"].astype(int)
        om = fit_model(model_name, Xot, yot)
        outer_train_score = om.predict_proba(Xot)[:,1]
        outer_val_score = om.predict_proba(Xov)[:,1]
        outer_rank = percentile_score(outer_val_score, outer_train_score)
        p = pd.DataFrame({"date": d.loc[ova,"date"].to_numpy(), "target": yov.to_numpy(), "rank_score": outer_rank, "raw_score": outer_val_score, "q": q, "fold_id": fid, "feature_set": set_name, "model": model_name})
        parts.append(p)
        fm = metric(yov, outer_rank >= q, outer_rank)
        fold_details.append({"fold_id":fid,"threshold_percentile":q,"inner_feasible_thresholds":nfeas,"inner_metrics":qm,"outer_metrics":fm})

    oof = pd.concat(parts, ignore_index=True)
    pred = oof.rank_score.to_numpy(float) >= oof.q.to_numpy(float)
    y = oof.target.to_numpy(int)
    m = metric(y, pred, oof.rank_score.to_numpy(float))
    m["worst_fold_recall"] = float(min(x["outer_metrics"]["recall"] for x in fold_details))
    m["max_fold_alert_rate"] = float(max(x["outer_metrics"]["alert_rate"] for x in fold_details))
    m["min_fold_npv"] = float(min(x["outer_metrics"]["green_npv"] for x in fold_details))
    return {"feature_set":set_name,"model":model_name,"feature_count":len(features),"metrics":m,"folds":fold_details,"median_q":float(oof.groupby("fold_id").q.first().median())}, oof


def selection_key(r):
    m = r["metrics"]
    stable = m["worst_fold_recall"] >= .50 and m["max_fold_alert_rate"] <= .50
    usable = m["recall"] >= .78 and m["green_npv"] >= .94 and m["alert_rate"] <= .42
    return (int(usable and stable), int(stable), m["f1"], m["pr_auc"], m["roc_auc"], m["precision"], -m["alert_rate"])


def main():
    d = pd.read_csv(PANEL, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if len(d) != 541:
        raise RuntimeError(f"Expected 541 rows, got {len(d)}")
    fs = feature_sets(list(d.columns))
    results=[]; all_oof=[]
    for set_name, features in fs.items():
        for model_name in ["logistic","extra_trees","hist_gb","xgb","catboost"]:
            r,o = evaluate_candidate(d, features, model_name, set_name)
            results.append(r); all_oof.append(o)
            print(set_name, model_name, json.dumps(r["metrics"]))
    selected = max(results, key=selection_key)
    selected_features = fs[selected["feature_set"]]
    pd.concat(all_oof, ignore_index=True).to_csv(OOF, index=False)

    v61 = json.loads(V61_REPORT.read_text(encoding="utf-8"))["selected_policy"]["metrics"]
    m = selected["metrics"]
    strict = {
        "roc_auc_ge_v6_1": m["roc_auc"] >= float(v61["roc_auc"]),
        "pr_auc_ge_v6_1": m["pr_auc"] >= float(v61["pr_auc"]),
        "recall_ge_v6_1_minus_2pp": m["recall"] >= float(v61["recall"])-.02,
        "precision_gt_v6_1": m["precision"] > float(v61["precision"]),
        "f1_gt_v6_1": m["f1"] > float(v61["f1"]),
        "green_npv_ge_095": m["green_npv"] >= .95,
        "alert_rate_le_v6_1": m["alert_rate"] <= float(v61["alert_rate"]),
        "worst_fold_recall_ge_060": m["worst_fold_recall"] >= .60,
        "fp_lt_v6_1": m["fp"] < int(v61["fp"]),
    }
    adopt = bool(all(strict.values()))

    # Final development artifact. Percentile threshold stores its training-score reference.
    Xfit, _, prep = prepare(d[selected_features], d[selected_features])
    yfit = d.target.astype(int)
    fitted = fit_model(selected["model"], Xfit, yfit)
    ref_scores = fitted.predict_proba(Xfit)[:,1]
    artifact = {
        "version": VERSION,
        "status": "DEVELOPMENT_ADOPTABLE_PENDING_EXTERNAL_VALIDATION" if adopt else "EXPERIMENTAL_NOT_ADOPTED",
        "feature_set": selected["feature_set"],
        "features": selected_features,
        "model_name": selected["model"],
        "model_object": fitted,
        "preprocessing": prep,
        "training_score_reference": ref_scores,
        "percentile_threshold": selected["median_q"],
        "red_supported": False,
    }
    joblib.dump(artifact, MODEL)

    report = {
        "version": VERSION,
        "status": artifact["status"],
        "scientific_boundary": "V7.4 is compact nested rolling-origin development evidence on the existing merchant panel. Feature sets are deterministic domain-selected compact subsets; thresholds are percentile-calibrated from past-only inner validation. No fresh external Saudi merchant validation has occurred.",
        "rows": len(d),
        "positive_rate": float(d.target.mean()),
        "feature_sets": {k:len(v) for k,v in fs.items()},
        "candidate_count": len(results),
        "selected": selected,
        "v6_1_reference": v61,
        "strict_adoption_gates": strict,
        "all_strict_adoption_gates_passed": adopt,
        "top_candidates": sorted(results, key=selection_key, reverse=True)[:8],
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = [
        "# Sales Sentinel V7.4 — Compact Percentile-Calibrated Temporal Model","",
        f"- Status: **{report['status']}**",
        f"- Selected: **{selected['feature_set']} / {selected['model']}**",
        f"- Features: **{selected['feature_count']}** (down from V7.1's 274)",
        f"- ROC-AUC: **{m['roc_auc']:.2%}**",
        f"- PR-AUC: **{m['pr_auc']:.2%}**",
        f"- Precision: **{m['precision']:.2%}**",
        f"- Recall: **{m['recall']:.2%}**",
        f"- F1: **{m['f1']:.2%}**",
        f"- GREEN NPV: **{m['green_npv']:.2%}**",
        f"- Alert rate: **{m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Max-fold alert rate: **{m['max_fold_alert_rate']:.2%}**",
        f"- V6.1 F1 / Recall / Precision: **{v61['f1']:.2%} / {v61['recall']:.2%} / {v61['precision']:.2%}**",
        f"- Adopt over V6.1: **{adopt}**",
        "- RED supported: **False**","",
        "Scientific boundary: development-only rolling-origin evidence; external real Saudi merchant validation remains required.",
    ]
    SUMMARY.write_text("\n".join(summary)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
