from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import train_merchant_market_fusion_v6_1 as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "merchant_market_fusion_v6_1"
MOD = ROOT / "models" / "merchant_market_fusion_v6_1"
DIAG = OUT / "oof_policy_diagnostics.csv"
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
MODEL = MOD / "merchant_market_fusion_v6_1.joblib"
VERSION = "SALES-SENTINEL-MERCHANT-MARKET-FUSION-6.1-REGIME-AWARE"

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


def regime_prediction(d: pd.DataFrame):
    mm = d["merchant_mean"].to_numpy(float)
    lr = d["merchant_logreg"].to_numpy(float)
    extra = d["merchant_extra"].to_numpy(float)
    disagree = d["merchant_disagreement"].to_numpy(float)
    market_p90 = d["market_v3__risk_p90"].to_numpy(float)

    strong_merchant = (
        (mm >= POLICY["strong_merchant_mean_min"])
        & (extra >= POLICY["strong_extra_min"])
    )
    quiet_market_asymmetric = (
        (market_p90 <= POLICY["quiet_market_risk_p90_max"])
        & (lr >= POLICY["quiet_logreg_min"])
        & (disagree >= POLICY["quiet_disagreement_min"])
        & (mm >= POLICY["quiet_merchant_mean_min"])
    )
    market_confirmed = (
        (market_p90 >= POLICY["market_confirm_risk_p90_min"])
        & (mm >= POLICY["market_confirm_merchant_mean_min"])
    )
    pred = strong_merchant | quiet_market_asymmetric | market_confirmed
    diagnostics = {
        "strong_merchant_alerts": int(strong_merchant.sum()),
        "quiet_market_asymmetric_alerts": int(quiet_market_asymmetric.sum()),
        "market_confirmed_alerts": int(market_confirmed.sum()),
        "final_alerts": int(pred.sum()),
    }
    return pred, diagnostics


def main():
    if not DIAG.exists():
        raise RuntimeError("Missing v6.1 OOF diagnostics; run diagnostic workflow first")
    if not MODEL.exists():
        raise RuntimeError("Missing v6.1 model artifact; run diagnostic workflow first")

    d = pd.read_csv(DIAG)
    y = d["y"].to_numpy(int)
    fold_ids = d["fold_id"].to_numpy(int)
    score = d["merchant_mean"].to_numpy(float)
    pred, policy_diag = regime_prediction(d)
    metrics, per_fold = base.evaluate_policy(y, score, pred, fold_ids)

    ranking = {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
    }

    supported = bool(
        metrics["recall"] >= 0.80
        and metrics["precision"] >= 0.32
        and metrics["f1"] >= 0.46
        and metrics["green_npv"] >= 0.95
        and metrics["alert_rate"] <= 0.43
        and metrics["worst_fold_recall"] >= 0.50
        and metrics["max_fold_alert_rate"] <= 0.60
    )

    v6_report = json.loads(
        (ROOT / "reports" / "merchant_market_fusion_v6" / "development_report.json").read_text(encoding="utf-8")
    )
    v6m = v6_report["selected_early"]["metrics"]

    v44_path = ROOT / "reports" / "merchant_total_triage_v4_4" / "development_report.json"
    v44 = json.loads(v44_path.read_text(encoding="utf-8"))
    v44m = ((v44.get("metrics") or {}).get("AMBER_or_RED_vs_15pct") or {})
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

    comparison_v6 = {
        "v6": v6m,
        "v6_1": metrics,
        "delta": {
            k: float(metrics[k] - v6m[k])
            for k in ["precision", "recall", "f1", "alert_rate", "green_npv"]
        },
        "false_positives_removed": int(v6m["fp"] - metrics["fp"]),
        "true_positives_changed": int(metrics["tp"] - v6m["tp"]),
        "missed_declines_changed": int(metrics["fn"] - v6m["fn"]),
    }
    comparison_v44 = {
        "v4_4": v44m,
        "v6_1": metrics,
        "delta": {
            k: float(metrics[k] - v44m[k])
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
        "regime_policy_supported": supported,
        "ranking_roc_auc": ranking["roc_auc"] >= contract["ranking_roc_auc_min"],
        "recall": metrics["recall"] >= contract["recall_min"],
        "precision": metrics["precision"] >= contract["precision_min"],
        "f1": metrics["f1"] >= contract["f1_min"],
        "green_npv": metrics["green_npv"] >= contract["green_npv_min"],
        "alert_rate": metrics["alert_rate"] <= contract["alert_rate_max"],
        "worst_fold_recall": metrics["worst_fold_recall"] >= contract["worst_fold_recall_min"],
        "max_fold_alert_rate": metrics["max_fold_alert_rate"] <= contract["max_fold_alert_rate_max"],
        "false_positives_reduced_vs_v6": metrics["fp"] < v6m["fp"],
        "tp_loss_vs_v6_bounded": metrics["tp"] >= v6m["tp"] - 5,
    }

    previous = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_FROZEN_PENDING_EXTERNAL_MERCHANT_VALIDATION",
        "scientific_boundary": (
            "V6.1 regime-aware policy is development evidence on the same 381 purged rolling-origin OOF merchant rows. "
            "The three decision branches use only merchant OOF scores and the prequential SAMA market channel available at prediction origin. "
            "Thresholds are rounded operational cutoffs selected on development OOF evidence; no independent real Saudi merchant longitudinal validation has yet occurred."
        ),
        "merchant_rows": previous.get("merchant_rows", 541),
        "merchant_raw_features": previous.get("merchant_raw_features", 391),
        "market_channel_features": previous.get("market_channel_features", 8),
        "oof_rows": int(len(d)),
        "early_positive_rate": float(y.mean()),
        "severe_positive_rate": previous.get("severe_positive_rate"),
        "folds": previous.get("folds", []),
        "market_snapshots": previous.get("market_snapshots", []),
        "ranking": ranking,
        "selected_policy": {
            "supported": supported,
            "selection_mode": "rounded_three_branch_regime_policy",
            "policy": POLICY,
            "logic": {
                "strong_merchant": "merchant_mean >= 0.70 AND ExtraTrees >= 0.60",
                "quiet_market_asymmetric": "SAMA market risk p90 <= 0.05 AND Logistic >= 0.45 AND merchant disagreement >= 0.10 AND merchant_mean >= 0.35",
                "market_confirmed": "SAMA market risk p90 >= 0.20 AND merchant_mean >= 0.35",
            },
            "metrics": metrics,
            "per_fold": per_fold,
            "diagnostics": policy_diag,
            "rounded_thresholds": True,
        },
        "red": previous.get("red", {}),
        "red_supported": False,
        "comparison_vs_v6": comparison_v6,
        "comparison_vs_v4_4": comparison_v44,
        "contract": contract,
        "gates": gates,
        "all_development_gates_passed": bool(all(gates.values())),
        "next_required_evidence": (
            "Freeze V6.1 parameters and validate on a never-used real Saudi merchant longitudinal time series before any production-performance claim."
        ),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    artifact = joblib.load(MODEL)
    artifact["version"] = VERSION
    artifact["status"] = report["status"]
    artifact["selected_early_channel"] = "merchant_regime_aware_with_prequential_sama"
    artifact["early_threshold"] = None
    artifact["decision_policy"] = {
        "type": "three_branch_regime_aware_precision_recovery",
        "parameters": POLICY,
        "logic": report["selected_policy"]["logic"],
        "requires_market_channel": True,
    }
    artifact["red_supported"] = False
    artifact["red_threshold"] = None
    joblib.dump(artifact, MODEL)

    m = metrics
    summary = [
        "# Sales Sentinel v6.1 — Regime-Aware Precision Recovery",
        "",
        "- Policy: **three-branch merchant + prequential SAMA regime policy**",
        f"- Merchant rows: **{report['merchant_rows']}**",
        f"- OOF rows: **{len(d)}**",
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
        "- RED supported: **False**",
        f"- Development gates: **{all(gates.values())}**",
    ]
    SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not all(gates.values()):
        raise SystemExit("V6.1 regime-aware policy did not pass all development gates")


if __name__ == "__main__":
    main()
