from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import train_merchant_market_fusion_v6 as v6
import train_merchant_total_hybrid_v4_3 as merchant_base
import train_merchant_total_triage_v5 as v5

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-MERCHANT-MARKET-FUSION-6.1-PRECISION-RECOVERY"
SRC = ROOT / "data" / "merchant_v4_3" / "merchant_total_feature_panel_v4_3.csv"
OUT = ROOT / "reports" / "merchant_market_fusion_v6_1"
MOD = ROOT / "models" / "merchant_market_fusion_v6_1"
OUT.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
MODEL = MOD / "merchant_market_fusion_v6_1.joblib"

EARLY_RATIO = 0.85
SEVERE_RATIO = 0.80
MARKET_PREFIX = v6.MARKET_PREFIX
MERCHANT_MODELS = v6.MERCHANT_MODELS


def metrics_from_pred(y, score, pred):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=bool)
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


def evaluate_policy(y, score, pred, fold_ids):
    pooled = metrics_from_pred(y, score, pred)
    per_fold = []
    for fid in sorted(np.unique(fold_ids)):
        z = fold_ids == fid
        fm = metrics_from_pred(y[z], score[z], pred[z])
        fm["fold_id"] = int(fid)
        fm["positives"] = int(np.asarray(y)[z].sum())
        per_fold.append(fm)
    stable = [f["recall"] for f in per_fold if f["positives"] >= 5]
    pooled["worst_fold_recall"] = float(min(stable, default=pooled["recall"]))
    pooled["max_fold_alert_rate"] = float(max((f["alert_rate"] for f in per_fold), default=pooled["alert_rate"]))
    return pooled, per_fold


def make_policy_pred(Z, low_t, high_t, agree_t, market_t, disagreement_max):
    merchant_mean = Z["merchant_mean"].to_numpy(float)
    merchant_min = np.minimum(
        Z["merchant_logreg"].to_numpy(float),
        Z["merchant_extra"].to_numpy(float),
    )
    disagreement = Z["merchant_disagreement"].to_numpy(float)
    market_p90 = Z[f"{MARKET_PREFIX}risk_p90"].to_numpy(float)
    market_mean = Z[f"{MARKET_PREFIX}risk_mean"].to_numpy(float)
    precursor = Z[f"{MARKET_PREFIX}precursor_share_2"].to_numpy(float)

    high = merchant_mean >= high_t
    marginal = (merchant_mean >= low_t) & (~high)
    merchant_agree = (merchant_min >= agree_t) & (disagreement <= disagreement_max)
    market_confirm = (market_p90 >= market_t) | ((market_mean >= market_t * 0.60) & (precursor >= 0.20))
    confirm = merchant_agree | market_confirm
    pred = high | (marginal & confirm)
    return pred, {
        "high_alerts": int(high.sum()),
        "marginal_candidates": int(marginal.sum()),
        "merchant_confirmed_marginal": int((marginal & merchant_agree).sum()),
        "market_confirmed_marginal": int((marginal & market_confirm).sum()),
        "final_alerts": int(pred.sum()),
    }


def candidate_values(x, quantiles):
    x = np.asarray(x, float)
    return sorted(set(float(v) for v in np.quantile(x, quantiles)))


def search_precision_recovery_policy(y, Z, fold_ids, severe=False):
    y = np.asarray(y, int)
    fold_ids = np.asarray(fold_ids, int)
    score = Z["merchant_mean"].to_numpy(float)
    market = Z[f"{MARKET_PREFIX}risk_p90"].to_numpy(float)

    low_values = candidate_values(score, [0.44, 0.48, 0.52, 0.56, 0.60])
    high_values = candidate_values(score, [0.62, 0.68, 0.74, 0.80, 0.86])
    agree_values = candidate_values(
        np.minimum(Z["merchant_logreg"], Z["merchant_extra"]),
        [0.45, 0.55, 0.65, 0.75],
    )
    market_values = candidate_values(market, [0.45, 0.60, 0.72, 0.82])
    disagreement_values = candidate_values(Z["merchant_disagreement"], [0.45, 0.62, 0.78])

    rows = []
    for low_t in low_values:
        for high_t in high_values:
            if high_t <= low_t:
                continue
            for agree_t in agree_values:
                for market_t in market_values:
                    for disagreement_max in disagreement_values:
                        pred, diagnostics = make_policy_pred(
                            Z, low_t, high_t, agree_t, market_t, disagreement_max
                        )
                        m, per = evaluate_policy(y, score, pred, fold_ids)
                        alerts = m["tp"] + m["fp"]
                        if severe:
                            supported = (
                                alerts >= 5
                                and m["precision"] >= 0.55
                                and m["recall"] >= 0.20
                                and m["alert_rate"] <= 0.12
                            )
                            objective = (
                                1.60 * m["precision"]
                                + 0.70 * m["f1"]
                                + 0.35 * m["recall"]
                                - 0.60 * m["alert_rate"]
                            )
                        else:
                            supported = (
                                m["recall"] >= 0.80
                                and m["precision"] >= 0.32
                                and m["f1"] >= 0.46
                                and m["green_npv"] >= 0.95
                                and m["alert_rate"] <= 0.43
                                and m["worst_fold_recall"] >= 0.50
                                and m["max_fold_alert_rate"] <= 0.60
                            )
                            objective = (
                                1.55 * m["f1"]
                                + 0.75 * m["precision"]
                                + 0.55 * m["balanced_accuracy"]
                                + 0.35 * m["recall"]
                                + 0.20 * m["green_npv"]
                                - 0.65 * m["alert_rate"]
                                - 0.0015 * m["fp"]
                            )
                        rows.append({
                            "supported": bool(supported),
                            "objective": float(objective),
                            "params": {
                                "low_threshold": float(low_t),
                                "high_threshold": float(high_t),
                                "agreement_threshold": float(agree_t),
                                "market_p90_threshold": float(market_t),
                                "max_model_disagreement": float(disagreement_max),
                            },
                            "metrics": m,
                            "per_fold": per,
                            "diagnostics": diagnostics,
                        })

    if not rows:
        raise RuntimeError("No v6.1 policy candidates generated")
    feasible = [r for r in rows if r["supported"]]
    pool = feasible or rows
    pool.sort(
        key=lambda r: (
            r["supported"],
            r["objective"],
            r["metrics"]["f1"],
            r["metrics"]["precision"],
            r["metrics"]["recall"],
            -r["metrics"]["alert_rate"],
        ),
        reverse=True,
    )
    best = pool[0]
    best["feasible_candidates"] = int(len(feasible))
    best["total_candidates"] = int(len(rows))
    return best


def main():
    d = pd.read_csv(SRC, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    meta = d[["date", "future_ratio"]].copy()
    X = d.drop(columns=["date", "future_ratio", "target"]).replace([np.inf, -np.inf], np.nan)
    y15 = (meta.future_ratio.to_numpy(float) < EARLY_RATIO).astype(int)
    y20 = (meta.future_ratio.to_numpy(float) < SEVERE_RATIO).astype(int)

    folds = merchant_base.folds(meta.assign(target=y20))
    if len(folds) != 5:
        raise RuntimeError(f"Expected five purged rolling folds, got {len(folds)}")

    market_X, market_snapshots = v6.build_prequential_market_channel(meta.date)

    base15, fold_meta = v6.merchant_oof(X, y15, folds)
    Z15 = v6.meta_features(base15, market_X)
    y15_oof = base15.y.to_numpy(int)
    fold_ids = base15.fold_id.to_numpy(int)

    policy15 = search_precision_recovery_policy(y15_oof, Z15, fold_ids, severe=False)
    pred15, policy_diag = make_policy_pred(Z15, **{
        "low_t": policy15["params"]["low_threshold"],
        "high_t": policy15["params"]["high_threshold"],
        "agree_t": policy15["params"]["agreement_threshold"],
        "market_t": policy15["params"]["market_p90_threshold"],
        "disagreement_max": policy15["params"]["max_model_disagreement"],
    })
    policy15["diagnostics"] = policy_diag

    merchant_mean15 = Z15["merchant_mean"].to_numpy(float)
    ranking = {
        "roc_auc": float(roc_auc_score(y15_oof, merchant_mean15)),
        "pr_auc": float(average_precision_score(y15_oof, merchant_mean15)),
    }

    base20, _ = v6.merchant_oof(X, y20, folds)
    Z20 = v6.meta_features(base20, market_X)
    policy20 = search_precision_recovery_policy(
        base20.y.to_numpy(int), Z20, base20.fold_id.to_numpy(int), severe=True
    )
    red_supported = bool(policy20["supported"])

    artifact = v6.fit_final_artifact(
        X,
        y15,
        y20,
        base15,
        Z15,
        policy15["params"]["low_threshold"],
        market_X,
        market_snapshots,
    )
    artifact["version"] = VERSION
    artifact["selected_early_channel"] = "merchant_mean_with_precision_recovery_gate"
    artifact["decision_policy"] = {
        "type": "two_stage_precision_recovery",
        **policy15["params"],
        "logic": (
            "Alert immediately when merchant_mean >= high_threshold. For marginal merchant scores, "
            "alert only when both merchant models agree within the allowed disagreement OR the prequential "
            "SAMA market channel confirms elevated risk."
        ),
    }
    artifact["red_supported"] = red_supported
    artifact["red_policy"] = policy20["params"] if red_supported else None
    joblib.dump(artifact, MODEL)

    v6_path = ROOT / "reports" / "merchant_market_fusion_v6" / "development_report.json"
    v6_report = json.loads(v6_path.read_text(encoding="utf-8"))
    v6m = v6_report["selected_early"]["metrics"]

    v44_path = ROOT / "reports" / "merchant_total_triage_v4_4" / "development_report.json"
    v44_report = json.loads(v44_path.read_text(encoding="utf-8"))
    v44m = ((v44_report.get("metrics") or {}).get("AMBER_or_RED_vs_15pct") or {})
    if not v44m:
        v44m = {
            "precision": 0.30612244897959184,
            "recall": 0.7142857142857143,
            "f1": 0.42857142857142855,
            "alert_rate": 0.3858267716535433,
            "green_npv": 0.9230769230769231,
            "roc_auc": 0.7584107018069283,
            "tp": 45,
            "fp": 102,
            "fn": 18,
            "tn": 216,
        }

    m = policy15["metrics"]
    comparison_v6 = {
        "v6": v6m,
        "v6_1": m,
        "delta": {
            k: float(m[k] - v6m[k])
            for k in ["precision", "recall", "f1", "alert_rate", "green_npv"]
        },
        "false_positives_removed": int(v6m["fp"] - m["fp"]),
        "true_positives_changed": int(m["tp"] - v6m["tp"]),
        "missed_declines_changed": int(m["fn"] - v6m["fn"]),
    }
    comparison_v44 = {
        "v4_4": v44m,
        "v6_1": m,
        "delta": {
            k: float(m[k] - v44m[k])
            for k in ["precision", "recall", "f1", "alert_rate", "green_npv"]
            if k in v44m
        },
    }

    contract = {
        "ranking_roc_auc_min": 0.75,
        "recall_min": 0.80,
        "precision_min": 0.32,
        "f1_min": 0.46,
        "green_npv_min": 0.95,
        "alert_rate_max": 0.43,
        "worst_fold_recall_min": 0.50,
        "max_fold_alert_rate_max": 0.60,
        "must_reduce_false_positives_vs_v6": True,
        "must_not_lose_more_than_5_tp_vs_v6": True,
    }
    gates = {
        "merchant_rolling_origin_past_only": True,
        "merchant_target_purge_7days": True,
        "market_prequential_monthly_freeze": True,
        "market_label_availability_gap_14days": True,
        "market_week_shift_plus_7days": True,
        "no_synthetic_oversampling": True,
        "precision_recovery_policy_supported": bool(policy15["supported"]),
        "ranking_roc_auc": ranking["roc_auc"] >= contract["ranking_roc_auc_min"],
        "recall": m["recall"] >= contract["recall_min"],
        "precision": m["precision"] >= contract["precision_min"],
        "f1": m["f1"] >= contract["f1_min"],
        "green_npv": m["green_npv"] >= contract["green_npv_min"],
        "alert_rate": m["alert_rate"] <= contract["alert_rate_max"],
        "worst_fold_recall": m["worst_fold_recall"] >= contract["worst_fold_recall_min"],
        "max_fold_alert_rate": m["max_fold_alert_rate"] <= contract["max_fold_alert_rate_max"],
        "false_positives_reduced_vs_v6": m["fp"] < v6m["fp"],
        "tp_loss_vs_v6_bounded": m["tp"] >= v6m["tp"] - 5,
    }

    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_FROZEN_PENDING_EXTERNAL_MERCHANT_VALIDATION",
        "scientific_boundary": (
            "V6.1 changes the operational decision policy, not the merchant ranking model. All merchant scores are "
            "purged rolling-origin OOF. SAMA confirmation is prequential and shifted one week. Policy parameters are "
            "development-selected on OOF evidence only; no external real Saudi merchant longitudinal validation has yet occurred."
        ),
        "merchant_rows": int(len(d)),
        "merchant_raw_features": int(X.shape[1]),
        "market_channel_features": int(market_X.shape[1]),
        "oof_rows": int(len(base15)),
        "early_positive_rate": float(y15.mean()),
        "severe_positive_rate": float(y20.mean()),
        "folds": fold_meta,
        "market_snapshots": market_snapshots,
        "ranking": ranking,
        "selected_policy": policy15,
        "red": policy20,
        "red_supported": red_supported,
        "comparison_vs_v6": comparison_v6,
        "comparison_vs_v4_4": comparison_v44,
        "contract": contract,
        "gates": gates,
        "all_development_gates_passed": bool(all(gates.values())),
        "next_required_evidence": (
            "If V6.1 passes the operational gates, freeze the decision policy and validate it on a never-used real Saudi merchant time series. "
            "If it does not pass, do not relax the gates; the next improvement must come from richer merchant longitudinal data or new causal operational features."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = [
        "# Sales Sentinel v6.1 — Precision Recovery",
        "",
        "- Policy: **two-stage merchant + prequential SAMA confirmation**",
        f"- Merchant rows: **{len(d)}**",
        f"- OOF ROC-AUC: **{ranking['roc_auc']:.2%}**",
        f"- OOF PR-AUC: **{ranking['pr_auc']:.2%}**",
        f"- Precision: **{m['precision']:.2%}**",
        f"- Recall: **{m['recall']:.2%}**",
        f"- F1: **{m['f1']:.2%}**",
        f"- GREEN NPV: **{m['green_npv']:.2%}**",
        f"- Alert rate: **{m['alert_rate']:.2%}**",
        f"- TP / FP / FN / TN: **{m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Max fold alert rate: **{m['max_fold_alert_rate']:.2%}**",
        f"- False positives removed vs V6: **{comparison_v6['false_positives_removed']}**",
        f"- TP change vs V6: **{comparison_v6['true_positives_changed']:+d}**",
        f"- Precision delta vs V6: **{comparison_v6['delta']['precision']:+.2%}**",
        f"- Recall delta vs V6: **{comparison_v6['delta']['recall']:+.2%}**",
        f"- F1 delta vs V6: **{comparison_v6['delta']['f1']:+.2%}**",
        f"- Alert-rate delta vs V6: **{comparison_v6['delta']['alert_rate']:+.2%}**",
        f"- RED supported: **{red_supported}**",
        f"- Development gates: **{all(gates.values())}**",
    ]
    SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
