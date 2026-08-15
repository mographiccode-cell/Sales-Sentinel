from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import finalize_merchant_market_fusion_v6_1 as v61

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V7.2-V6.1-PLUS-V7.1-RANK-FUSION"
V61_DIAG = ROOT / "reports" / "merchant_market_fusion_v6_1" / "oof_policy_diagnostics.csv"
V61_REPORT = ROOT / "reports" / "merchant_market_fusion_v6_1" / "development_report.json"
V71_OOF = ROOT / "reports" / "merchant_category_signals_v7_1" / "oof_predictions.csv"
V71_REPORT = ROOT / "reports" / "merchant_category_signals_v7_1" / "development_report.json"
OUT = ROOT / "reports" / "merchant_fusion_v7_2"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
FUSION_OOF = OUT / "oof_fusion_predictions.csv"


def evaluate(y: np.ndarray, pred: np.ndarray, folds: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=bool)
    per_fold = []
    for fid in sorted(np.unique(folds)):
        ix = folds == fid
        yy = y[ix]
        pp = pred[ix]
        tp = int(((yy == 1) & pp).sum())
        fp = int(((yy == 0) & pp).sum())
        fn = int(((yy == 1) & (~pp)).sum())
        tn = int(((yy == 0) & (~pp)).sum())
        per_fold.append({
            "fold_id": int(fid),
            "rows": int(ix.sum()),
            "positives": int((yy == 1).sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(tp / max(tp + fp, 1)),
            "recall": float(tp / max(tp + fn, 1)),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "alert_rate": float(pp.mean()),
            "green_npv": float(tn / max(tn + fn, 1)),
        })

    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & (~pred)).sum())
    tn = int(((y == 0) & (~pred)).sum())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "alert_rate": float(pred.mean()),
        "green_npv": float(tn / max(tn + fn, 1)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "worst_fold_recall": float(min(x["recall"] for x in per_fold)),
        "max_fold_alert_rate": float(max(x["alert_rate"] for x in per_fold)),
        "per_fold": per_fold,
    }


def load_aligned() -> pd.DataFrame:
    d = pd.read_csv(V61_DIAG)
    q = pd.read_csv(V71_OOF, parse_dates=["date"])
    q = q[(q["scope"] == "merchant_plus_category_signals") & (q["model"] == "mean_ensemble")].copy()
    q = q.sort_values(["fold_id", "date"]).reset_index(drop=True)
    d = d.copy()
    d["row_in_fold"] = d.groupby("fold_id").cumcount()
    q["row_in_fold"] = q.groupby("fold_id").cumcount()
    keep = q[["date", "target", "score", "threshold", "fold_id", "row_in_fold"]].rename(
        columns={"score": "v71_rank_score", "threshold": "v71_nested_threshold"}
    )
    merged = d.merge(keep, on=["fold_id", "row_in_fold"], how="inner", validate="one_to_one")
    if len(merged) != len(d) or len(merged) != 381:
        raise RuntimeError(f"V6.1/V7.1 OOF alignment failed: {len(merged)} vs {len(d)}")
    if not np.array_equal(merged["y"].to_numpy(int), merged["target"].to_numpy(int)):
        bad = int((merged["y"].to_numpy(int) != merged["target"].to_numpy(int)).sum())
        raise RuntimeError(f"V6.1/V7.1 target sequence mismatch: {bad} rows")
    return merged.sort_values(["fold_id", "row_in_fold"]).reset_index(drop=True)


def make_prediction(d: pd.DataFrame, base_pred: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = d["v71_rank_score"].to_numpy(float)
    mm = d["merchant_mean"].to_numpy(float)
    lr = d["merchant_logreg"].to_numpy(float)
    market = d["market_v3__risk_p90"].to_numpy(float)

    veto = (
        base_pred
        & (score < params["veto_v71_max"])
        & (mm < params["veto_merchant_mean_max"])
        & (market < 0.20)
    )
    rescue = (
        (~base_pred)
        & (score >= params["rescue_v71_min"])
        & (mm >= params["rescue_merchant_mean_min"])
        & ((lr >= 0.45) | (market >= 0.20))
    )
    pred = (base_pred & (~veto)) | rescue
    return pred, veto, rescue


def candidate_key(item: dict) -> tuple:
    m = item["metrics"]
    return (m["f1"], m["precision"], -m["fp"], m["recall"], -m["alert_rate"])


def main():
    for p in [V61_DIAG, V61_REPORT, V71_OOF, V71_REPORT]:
        if not p.exists():
            raise RuntimeError(f"Required evidence missing: {p}")

    d = load_aligned()
    y = d["y"].to_numpy(int)
    folds = d["fold_id"].to_numpy(int)
    base_pred, base_diag = v61.regime_prediction(d)
    base_metrics = evaluate(y, base_pred, folds)

    v61_raw = json.loads(V61_REPORT.read_text(encoding="utf-8"))
    v71_raw = json.loads(V71_REPORT.read_text(encoding="utf-8"))

    v61_score = d["merchant_mean"].to_numpy(float)
    v71_score = d["v71_rank_score"].to_numpy(float)
    ranking = {
        "v6_1_merchant_mean": {
            "roc_auc": float(roc_auc_score(y, v61_score)),
            "pr_auc": float(average_precision_score(y, v61_score)),
        },
        "v7_1_category_ensemble": {
            "roc_auc": float(roc_auc_score(y, v71_score)),
            "pr_auc": float(average_precision_score(y, v71_score)),
        },
    }
    blends = []
    for w in [0.00, 0.25, 0.50, 0.75, 1.00]:
        # Rank normalization makes the two OOF score scales comparable without using labels.
        r61 = pd.Series(v61_score).rank(pct=True).to_numpy(float)
        r71 = pd.Series(v71_score).rank(pct=True).to_numpy(float)
        s = (1.0 - w) * r61 + w * r71
        blends.append({
            "v71_weight": w,
            "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
        })
    best_blend = max(blends, key=lambda z: (z["pr_auc"], z["roc_auc"]))
    ranking["rank_blends"] = blends
    ranking["best_rank_blend"] = best_blend

    veto_scores = [-1.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    veto_mm = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
    rescue_scores = [1.01, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    rescue_mm = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    strict = []
    contract_pool = []
    all_candidates = []
    for vt, vm, rt, rm in product(veto_scores, veto_mm, rescue_scores, rescue_mm):
        params = {
            "veto_v71_max": float(vt),
            "veto_merchant_mean_max": float(vm),
            "rescue_v71_min": float(rt),
            "rescue_merchant_mean_min": float(rm),
        }
        pred, veto, rescue = make_prediction(d, base_pred, params)
        m = evaluate(y, pred, folds)
        item = {
            "parameters": params,
            "metrics": m,
            "vetoed_alerts": int(veto.sum()),
            "rescued_alerts": int(rescue.sum()),
        }
        all_candidates.append(item)

        # Strict target: do not lose a V6.1 true positive or fold stability while reducing false alarms.
        if (
            m["tp"] >= base_metrics["tp"]
            and m["fp"] < base_metrics["fp"]
            and m["f1"] > base_metrics["f1"]
            and m["green_npv"] >= 0.95
            and m["worst_fold_recall"] >= base_metrics["worst_fold_recall"]
            and m["max_fold_alert_rate"] <= base_metrics["max_fold_alert_rate"] + 1e-12
            and m["alert_rate"] <= base_metrics["alert_rate"]
        ):
            strict.append(item)

        # Minimum academic/operational contract if strict dominance is impossible.
        if (
            m["recall"] >= 0.80
            and m["precision"] >= base_metrics["precision"]
            and m["f1"] >= base_metrics["f1"]
            and m["green_npv"] >= 0.95
            and m["alert_rate"] <= base_metrics["alert_rate"]
            and m["worst_fold_recall"] >= 0.60
            and m["max_fold_alert_rate"] <= 0.43
            and m["fp"] < base_metrics["fp"]
            and m["tp"] >= base_metrics["tp"] - 1
        ):
            contract_pool.append(item)

    if strict:
        selected = max(strict, key=candidate_key)
        selection_mode = "strict_dominance_over_v6_1"
    elif contract_pool:
        selected = max(contract_pool, key=candidate_key)
        selection_mode = "bounded_tradeoff_contract"
    else:
        selected = max(all_candidates, key=candidate_key)
        selection_mode = "best_experimental_candidate_contract_failed"

    params = selected["parameters"]
    selected_pred, selected_veto, selected_rescue = make_prediction(d, base_pred, params)
    selected_metrics = evaluate(y, selected_pred, folds)

    # Local perturbation test: policy should not work at only one exact threshold combination.
    robust = []
    vt_vals = sorted(set([params["veto_v71_max"], params["veto_v71_max"] - 0.02, params["veto_v71_max"] + 0.02]))
    vm_vals = sorted(set([params["veto_merchant_mean_max"], params["veto_merchant_mean_max"] - 0.05, params["veto_merchant_mean_max"] + 0.05]))
    rt_vals = sorted(set([params["rescue_v71_min"], params["rescue_v71_min"] - 0.02, params["rescue_v71_min"] + 0.02]))
    rm_vals = sorted(set([params["rescue_merchant_mean_min"], params["rescue_merchant_mean_min"] - 0.05, params["rescue_merchant_mean_min"] + 0.05]))
    for vt, vm, rt, rm in product(vt_vals, vm_vals, rt_vals, rm_vals):
        pp = {
            "veto_v71_max": float(vt),
            "veto_merchant_mean_max": float(vm),
            "rescue_v71_min": float(rt),
            "rescue_merchant_mean_min": float(rm),
        }
        pred, _, _ = make_prediction(d, base_pred, pp)
        m = evaluate(y, pred, folds)
        passes = bool(
            m["recall"] >= 0.80
            and m["green_npv"] >= 0.95
            and m["worst_fold_recall"] >= 0.50
            and m["alert_rate"] <= 0.43
            and m["fp"] < base_metrics["fp"]
            and m["tp"] >= base_metrics["tp"] - 2
        )
        robust.append({"parameters": pp, "passes": passes, "metrics": m})
    robust_pass = sum(int(x["passes"]) for x in robust)

    gates = {
        "oof_alignment_exact": True,
        "v61_policy_frozen_as_base": True,
        "v71_channel_is_nested_oof_ranking": True,
        "no_new_blind_holdout_claim": True,
        "no_smote_or_synthetic_oversampling": True,
        "recall_ge_080": selected_metrics["recall"] >= 0.80,
        "precision_ge_v61": selected_metrics["precision"] >= base_metrics["precision"],
        "f1_gt_v61": selected_metrics["f1"] > base_metrics["f1"],
        "green_npv_ge_095": selected_metrics["green_npv"] >= 0.95,
        "alert_rate_le_v61": selected_metrics["alert_rate"] <= base_metrics["alert_rate"],
        "false_positives_lt_v61": selected_metrics["fp"] < base_metrics["fp"],
        "tp_loss_max_1": selected_metrics["tp"] >= base_metrics["tp"] - 1,
        "worst_fold_recall_ge_060": selected_metrics["worst_fold_recall"] >= 0.60,
        "max_fold_alert_rate_le_043": selected_metrics["max_fold_alert_rate"] <= 0.43,
        "local_perturbation_support": robust_pass >= max(10, int(0.20 * len(robust))),
    }
    adopt = bool(all(gates.values()))

    out = d.copy()
    out["v61_alert"] = base_pred.astype(int)
    out["v72_veto"] = selected_veto.astype(int)
    out["v72_rescue"] = selected_rescue.astype(int)
    out["v72_alert"] = selected_pred.astype(int)
    out.to_csv(FUSION_OOF, index=False)

    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_FUSION_ACCEPTED_PENDING_EXTERNAL_VALIDATION" if adopt else "EXPERIMENTAL_FUSION_NOT_ADOPTED",
        "scientific_boundary": (
            "V7.2 is a constrained decision-fusion development experiment on the same 381 purged rolling-origin OOF rows already used during V6.1/V7.1 development. "
            "V6.1 is kept frozen as the base decision policy. The V7.1 merchant-plus-category mean-ensemble OOF ranking score is used only to veto weak non-market-confirmed V6.1 alerts and rescue high-confidence non-alerts. "
            "This is not independent validation and must not be described as production performance."
        ),
        "rows": int(len(d)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "alignment": {
            "method": "fold_id + chronological row_in_fold",
            "target_sequence_exact_match": True,
        },
        "ranking": ranking,
        "base_v6_1": {
            "metrics": base_metrics,
            "diagnostics": base_diag,
            "report_metrics": v61_raw.get("selected_policy", {}).get("metrics", {}),
        },
        "v7_1_reference": {
            "version": v71_raw.get("version"),
            "ablation": v71_raw.get("ablation", {}),
        },
        "search": {
            "candidate_count": len(all_candidates),
            "strict_dominance_candidates": len(strict),
            "bounded_contract_candidates": len(contract_pool),
            "selection_mode": selection_mode,
        },
        "selected_policy": {
            "logic": {
                "base": "start from frozen V6.1 three-branch regime policy",
                "veto": "remove a V6.1 alert only when V7.1 rank score is below veto_v71_max, merchant_mean is below veto_merchant_mean_max, and SAMA risk p90 < 0.20",
                "rescue": "add a non-alert only when V7.1 rank score >= rescue_v71_min, merchant_mean >= rescue_merchant_mean_min, and (merchant Logistic >= 0.45 OR SAMA risk p90 >= 0.20)",
            },
            "parameters": params,
            "vetoed_alerts": int(selected_veto.sum()),
            "rescued_alerts": int(selected_rescue.sum()),
            "metrics": selected_metrics,
        },
        "comparison_vs_v6_1": {
            "precision_delta": float(selected_metrics["precision"] - base_metrics["precision"]),
            "recall_delta": float(selected_metrics["recall"] - base_metrics["recall"]),
            "f1_delta": float(selected_metrics["f1"] - base_metrics["f1"]),
            "npv_delta": float(selected_metrics["green_npv"] - base_metrics["green_npv"]),
            "alert_rate_delta": float(selected_metrics["alert_rate"] - base_metrics["alert_rate"]),
            "fp_removed": int(base_metrics["fp"] - selected_metrics["fp"]),
            "tp_change": int(selected_metrics["tp"] - base_metrics["tp"]),
        },
        "robustness": {
            "local_perturbations": len(robust),
            "passing_perturbations": robust_pass,
            "passing_share": float(robust_pass / max(len(robust), 1)),
        },
        "gates": gates,
        "all_development_gates_passed": adopt,
        "adopt_over_v6_1": adopt,
        "red_supported": False,
        "next_required_evidence": "Independent real Saudi merchant longitudinal validation on never-used data.",
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    m = selected_metrics
    b = base_metrics
    summary = [
        "# Sales Sentinel V7.2 — V6.1 + V7.1 Ranking Fusion",
        "",
        f"- Selection mode: **{selection_mode}**",
        f"- Development rows: **{len(d)}**",
        f"- Positives: **{int(y.sum())}**",
        f"- V7.1 category-ensemble ROC-AUC: **{ranking['v7_1_category_ensemble']['roc_auc']:.2%}**",
        f"- V7.1 category-ensemble PR-AUC: **{ranking['v7_1_category_ensemble']['pr_auc']:.2%}**",
        f"- Best rank-blend ROC-AUC: **{best_blend['roc_auc']:.2%}**",
        f"- Best rank-blend PR-AUC: **{best_blend['pr_auc']:.2%}**",
        "",
        "## Operational comparison",
        f"- V6.1 precision: **{b['precision']:.2%}** -> V7.2 **{m['precision']:.2%}**",
        f"- V6.1 recall: **{b['recall']:.2%}** -> V7.2 **{m['recall']:.2%}**",
        f"- V6.1 F1: **{b['f1']:.2%}** -> V7.2 **{m['f1']:.2%}**",
        f"- V6.1 GREEN NPV: **{b['green_npv']:.2%}** -> V7.2 **{m['green_npv']:.2%}**",
        f"- V6.1 alert rate: **{b['alert_rate']:.2%}** -> V7.2 **{m['alert_rate']:.2%}**",
        f"- V6.1 TP/FP/FN/TN: **{b['tp']}/{b['fp']}/{b['fn']}/{b['tn']}**",
        f"- V7.2 TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- FP removed: **{b['fp'] - m['fp']}**",
        f"- TP change: **{m['tp'] - b['tp']:+d}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Max-fold alert rate: **{m['max_fold_alert_rate']:.2%}**",
        f"- Vetoed alerts: **{int(selected_veto.sum())}**",
        f"- Rescued alerts: **{int(selected_rescue.sum())}**",
        f"- Robust perturbations passing: **{robust_pass}/{len(robust)}**",
        f"- Development gates passed: **{adopt}**",
        f"- Adopt over V6.1: **{adopt}**",
        "- RED supported: **False**",
        "",
        "Scientific boundary: this is development fusion on previously used OOF evidence, not a new blind or external validation set.",
    ]
    SUMMARY.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
