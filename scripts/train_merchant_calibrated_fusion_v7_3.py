from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V7.3-PREQUENTIAL-CALIBRATED-FUSION"
V61_DIAG = ROOT / "reports" / "merchant_market_fusion_v6_1" / "oof_policy_diagnostics.csv"
V61_REPORT = ROOT / "reports" / "merchant_market_fusion_v6_1" / "development_report.json"
V71_OOF = ROOT / "reports" / "merchant_category_signals_v7_1" / "oof_predictions.csv"
OUT = ROOT / "reports" / "merchant_calibrated_fusion_v7_3"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_predictions.csv"

POLICY = {
    "strong_merchant_mean_min": 0.70,
    "strong_extra_min": 0.60,
    "quiet_market_risk_p90_max": 0.05,
    "quiet_logreg_min": 0.45,
    "quiet_disagreement_min": 0.10,
    "quiet_merchant_mean_min": 0.35,
    "market_confirm_risk_p90_min": 0.20,
    "market_confirm_merchant_mean_min": 0.35,
}


def v61_pred(d: pd.DataFrame) -> np.ndarray:
    mm = d.merchant_mean.to_numpy(float)
    lr = d.merchant_logreg.to_numpy(float)
    ex = d.merchant_extra.to_numpy(float)
    dis = d.merchant_disagreement.to_numpy(float)
    p90 = d.market_v3__risk_p90.to_numpy(float)
    strong = (mm >= .70) & (ex >= .60)
    quiet = (p90 <= .05) & (lr >= .45) & (dis >= .10) & (mm >= .35)
    market = (p90 >= .20) & (mm >= .35)
    return strong | quiet | market


def evaluate(y, pred, folds):
    y = np.asarray(y, int); pred = np.asarray(pred, bool); folds = np.asarray(folds, int)
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum()); tn = int(((y == 0) & ~pred).sum())
    per = []
    for fid in sorted(np.unique(folds)):
        ix = folds == fid; yy = y[ix]; pp = pred[ix]
        tp0 = int(((yy == 1) & pp).sum()); fp0 = int(((yy == 0) & pp).sum())
        fn0 = int(((yy == 1) & ~pp).sum()); tn0 = int(((yy == 0) & ~pp).sum())
        per.append({
            "fold_id": int(fid), "tp": tp0, "fp": fp0, "fn": fn0, "tn": tn0,
            "precision": tp0 / max(tp0 + fp0, 1), "recall": tp0 / max(tp0 + fn0, 1),
            "f1": 2 * tp0 / max(2 * tp0 + fp0 + fn0, 1), "alert_rate": float(pp.mean()),
            "green_npv": tn0 / max(tn0 + fn0, 1)
        })
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alert_rate": float(pred.mean()), "green_npv": tn / max(tn + fn, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "worst_fold_recall": float(min(x["recall"] for x in per)),
        "max_fold_alert_rate": float(max(x["alert_rate"] for x in per)),
        "per_fold": per,
    }


def choose_threshold(y, score):
    rows = []
    for t in np.unique(np.r_[np.linspace(.05, .95, 181), np.quantile(score, np.linspace(.05, .95, 91))]):
        pred = score >= t
        m = evaluate(y, pred, np.zeros(len(y), dtype=int))
        feasible = m["recall"] >= .80 and m["green_npv"] >= .95 and m["alert_rate"] <= .40
        objective = m["f1"] + .20 * m["precision"] + .10 * m["balanced_accuracy"] - .30 * max(m["alert_rate"] - .35, 0)
        rows.append((int(feasible), objective, float(t), m))
    rows.sort(key=lambda z: (z[0], z[1]), reverse=True)
    return rows[0][2], rows[0][3], bool(rows[0][0])


def empirical_percentile(history, values):
    h = np.sort(np.asarray(history, float))
    if len(h) == 0:
        return np.full(len(values), .5)
    return np.searchsorted(h, np.asarray(values, float), side="right") / len(h)


def load_aligned():
    a = pd.read_csv(V61_DIAG).reset_index(drop=True)
    z = pd.read_csv(V71_OOF, parse_dates=["date"])
    ens = z[(z.scope == "merchant_plus_category_signals") & (z.model == "mean_ensemble")].sort_values(["fold_id", "date"]).reset_index(drop=True)
    hist = z[(z.scope == "merchant_plus_category_signals") & (z.model == "hist_gb")].sort_values(["fold_id", "date"]).reset_index(drop=True)
    if not (len(a) == len(ens) == len(hist) == 381):
        raise RuntimeError(f"Alignment rows mismatch: {len(a)}, {len(ens)}, {len(hist)}")
    if not np.array_equal(a.y.to_numpy(int), ens.target.to_numpy(int)):
        raise RuntimeError("V6.1 and V7.1 target order mismatch")
    if not np.array_equal(a.fold_id.to_numpy(int), ens.fold_id.to_numpy(int)):
        raise RuntimeError("V6.1 and V7.1 fold order mismatch")
    a["date"] = ens.date
    a["v71_ensemble"] = ens.score.to_numpy(float)
    a["v71_hist_gb"] = hist.score.to_numpy(float)
    return a


def main():
    d = load_aligned()
    y = d.y.to_numpy(int); folds = d.fold_id.to_numpy(int)
    base = v61_pred(d)
    base_m = evaluate(y, base, folds)
    v61_report = json.loads(V61_REPORT.read_text(encoding="utf-8"))

    features = [
        "merchant_logreg", "merchant_extra", "merchant_mean", "merchant_disagreement",
        "market_v3__risk_mean", "market_v3__risk_max", "market_v3__risk_p90",
        "market_v3__risk_share_25", "market_v3__precursor_mean",
        "v71_ensemble", "v71_hist_gb",
    ]

    policy_preds = {"meta_logistic": np.zeros(len(d), bool), "causal_rank": np.zeros(len(d), bool), "guarded_meta": np.zeros(len(d), bool)}
    policy_scores = {k: np.full(len(d), np.nan) for k in policy_preds}
    fold_details = []

    for fid in sorted(np.unique(folds)):
        cur = folds == fid
        if fid == 0:
            for k in policy_preds:
                policy_preds[k][cur] = base[cur]
                policy_scores[k][cur] = d.loc[cur, "merchant_mean"].to_numpy(float)
            fold_details.append({"fold_id": int(fid), "mode": "v6_1_bootstrap_no_prior_oof"})
            continue

        hist_mask = folds < fid
        Xh = d.loc[hist_mask, features].replace([np.inf, -np.inf], np.nan).copy()
        Xc = d.loc[cur, features].replace([np.inf, -np.inf], np.nan).copy()
        med = Xh.median(numeric_only=True)
        Xh = Xh.fillna(med).fillna(0.0); Xc = Xc.fillna(med).fillna(0.0)
        yh = y[hist_mask]

        model = make_pipeline(StandardScaler(), LogisticRegression(C=.10, class_weight="balanced", max_iter=5000, random_state=42))
        model.fit(Xh, yh)
        sh = model.predict_proba(Xh)[:, 1]
        sc = model.predict_proba(Xc)[:, 1]
        t_meta, meta_hist_m, meta_feasible = choose_threshold(yh, sh)
        meta_pred = sc >= t_meta

        # Causal percentile blend: ranking signal is normalized only against earlier OOF rows.
        rh_mm = empirical_percentile(d.loc[hist_mask, "merchant_mean"], d.loc[hist_mask, "merchant_mean"])
        rh_v7 = empirical_percentile(d.loc[hist_mask, "v71_ensemble"], d.loc[hist_mask, "v71_ensemble"])
        rc_mm = empirical_percentile(d.loc[hist_mask, "merchant_mean"], d.loc[cur, "merchant_mean"])
        rc_v7 = empirical_percentile(d.loc[hist_mask, "v71_ensemble"], d.loc[cur, "v71_ensemble"])
        blend_h = .25 * rh_mm + .75 * rh_v7
        blend_c = .25 * rc_mm + .75 * rc_v7
        t_rank, rank_hist_m, rank_feasible = choose_threshold(yh, blend_h)
        rank_pred = blend_c >= t_rank

        # Guard only with the strongest V6.1 merchant-consensus branch to recover obvious misses.
        strong_cur = (d.loc[cur, "merchant_mean"].to_numpy(float) >= .70) & (d.loc[cur, "merchant_extra"].to_numpy(float) >= .60)
        guarded = meta_pred | strong_cur

        policy_preds["meta_logistic"][cur] = meta_pred
        policy_preds["causal_rank"][cur] = rank_pred
        policy_preds["guarded_meta"][cur] = guarded
        policy_scores["meta_logistic"][cur] = sc
        policy_scores["causal_rank"][cur] = blend_c
        policy_scores["guarded_meta"][cur] = sc
        fold_details.append({
            "fold_id": int(fid), "history_rows": int(hist_mask.sum()), "validation_rows": int(cur.sum()),
            "meta_threshold": float(t_meta), "meta_history_feasible": meta_feasible,
            "rank_threshold": float(t_rank), "rank_history_feasible": rank_feasible,
            "meta_history_metrics": meta_hist_m, "rank_history_metrics": rank_hist_m,
        })

    results = {}
    for name, pred in policy_preds.items():
        m = evaluate(y, pred, folds)
        score = policy_scores[name]
        ok = np.isfinite(score)
        m["roc_auc"] = float(roc_auc_score(y[ok], score[ok]))
        m["pr_auc"] = float(average_precision_score(y[ok], score[ok]))
        results[name] = m

    contract = lambda m: (
        m["recall"] >= .80 and m["green_npv"] >= .95 and m["worst_fold_recall"] >= .60
        and m["alert_rate"] <= base_m["alert_rate"] and m["fp"] < base_m["fp"] and m["f1"] > base_m["f1"]
    )
    passing = [(k, m) for k, m in results.items() if contract(m)]
    if passing:
        selected_name, selected = max(passing, key=lambda x: (x[1]["f1"], x[1]["precision"], -x[1]["fp"]))
        status = "DEVELOPMENT_CANDIDATE_BEATS_V6_1_REQUIRES_FRESH_VALIDATION"
        adopt = True
    else:
        selected_name, selected = max(results.items(), key=lambda x: (x[1]["f1"], x[1]["recall"], x[1]["green_npv"]))
        status = "EXPERIMENTAL_NOT_ADOPTED_V6_1_REMAINS_BEST_OPERATIONAL"
        adopt = False

    out = d[["date", "y", "fold_id", "merchant_mean", "v71_ensemble", "v71_hist_gb"]].copy()
    out["v6_1_pred"] = base.astype(int)
    for k in policy_preds:
        out[f"{k}_score"] = policy_scores[k]
        out[f"{k}_pred"] = policy_preds[k].astype(int)
    out.to_csv(OOF, index=False)

    report = {
        "version": VERSION, "status": status,
        "scientific_boundary": "V7.3 is development-only prequential calibration on previously used OOF evidence. Fold 0 bootstraps from frozen V6.1 because no earlier OOF exists; folds 1-4 use only earlier folds to fit/calibrate. Any superiority must be confirmed on fresh never-used merchant data.",
        "rows": int(len(d)), "positives": int(y.sum()),
        "base_v6_1": base_m,
        "base_v6_1_report_metrics": v61_report["selected_policy"]["metrics"],
        "features": features, "fold_details": fold_details,
        "candidates": results,
        "selected": {"policy": selected_name, "metrics": selected, "passes_strict_dominance_contract": adopt},
        "strict_contract": {"recall_min": .80, "green_npv_min": .95, "worst_fold_recall_min": .60, "alert_rate_not_above_v6_1": True, "fp_strictly_below_v6_1": True, "f1_strictly_above_v6_1": True},
        "external_real_saudi_validation": "Pending",
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    s = selected; b = base_m
    SUMMARY.write_text("\n".join([
        "# Sales Sentinel V7.3 — Prequential Calibrated Fusion",
        "",
        f"- Status: **{status}**",
        f"- Selected policy: **{selected_name}**",
        f"- Development rows: **{len(d)}**",
        f"- V6.1 Precision / Recall / F1: **{b['precision']:.2%} / {b['recall']:.2%} / {b['f1']:.2%}**",
        f"- V7.3 Precision / Recall / F1: **{s['precision']:.2%} / {s['recall']:.2%} / {s['f1']:.2%}**",
        f"- V6.1 NPV / Alert rate: **{b['green_npv']:.2%} / {b['alert_rate']:.2%}**",
        f"- V7.3 NPV / Alert rate: **{s['green_npv']:.2%} / {s['alert_rate']:.2%}**",
        f"- V6.1 TP/FP/FN/TN: **{b['tp']}/{b['fp']}/{b['fn']}/{b['tn']}**",
        f"- V7.3 TP/FP/FN/TN: **{s['tp']}/{s['fp']}/{s['fn']}/{s['tn']}**",
        f"- V7.3 ROC-AUC / PR-AUC: **{s['roc_auc']:.2%} / {s['pr_auc']:.2%}**",
        f"- Worst-fold recall: **{s['worst_fold_recall']:.2%}**",
        f"- Strict dominance contract passed: **{adopt}**",
        "",
        "Scientific boundary: development-only prequential evidence; fresh external merchant validation is required before any production claim.",
    ]) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
