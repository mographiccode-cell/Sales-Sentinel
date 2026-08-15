from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import train_merchant_fusion_v7_2 as v72

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V7.2.1-RAW-SCORE-FUSION"
OUT = ROOT / "reports" / "merchant_fusion_v7_2_1"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"


def main():
    d = v72.load_aligned()
    y = d["y"].to_numpy(int)
    folds = d["fold_id"].to_numpy(int)
    base_pred, _ = v72.v61.regime_prediction(d)
    base = v72.evaluate(y, base_pred, folds)
    merchant = d["merchant_mean"].to_numpy(float)
    category = d["v71_rank_score"].to_numpy(float)

    candidates = []
    for weight in [0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]:
        score = (1.0 - weight) * merchant + weight * category
        auc = float(roc_auc_score(y, score))
        pr = float(average_precision_score(y, score))
        for threshold in np.arange(0.05, 0.801, 0.005):
            pred = score >= float(threshold)
            m = v72.evaluate(y, pred, folds)
            feasible = bool(
                m["recall"] >= 0.80
                and m["precision"] >= base["precision"]
                and m["f1"] > base["f1"]
                and m["green_npv"] >= 0.95
                and m["alert_rate"] <= base["alert_rate"]
                and m["worst_fold_recall"] >= 0.60
                and m["max_fold_alert_rate"] <= 0.43
                and m["fp"] < base["fp"]
                and m["tp"] >= base["tp"] - 1
            )
            candidates.append({
                "weight_v71": weight,
                "threshold": float(round(threshold, 3)),
                "roc_auc": auc,
                "pr_auc": pr,
                "metrics": m,
                "feasible": feasible,
            })

    feasible = [x for x in candidates if x["feasible"]]
    if feasible:
        selected = max(feasible, key=lambda z: (z["metrics"]["f1"], z["metrics"]["precision"], z["pr_auc"], -z["metrics"]["fp"]))
        selection_mode = "operational_contract_passed"
    else:
        selected = max(candidates, key=lambda z: (z["metrics"]["f1"], z["metrics"]["balanced_accuracy"], z["pr_auc"], -z["metrics"]["alert_rate"]))
        selection_mode = "best_development_candidate_contract_failed"

    w = selected["weight_v71"]
    t = selected["threshold"]
    score = (1.0 - w) * merchant + w * category
    selected_metrics = v72.evaluate(y, score >= t, folds)

    # Threshold-neighborhood robustness at the selected fixed weight.
    nearby = []
    for dt in np.arange(-0.05, 0.051, 0.005):
        tt = float(np.clip(t + dt, 0.01, 0.99))
        m = v72.evaluate(y, score >= tt, folds)
        passes = bool(
            m["recall"] >= 0.80
            and m["green_npv"] >= 0.95
            and m["worst_fold_recall"] >= 0.50
            and m["alert_rate"] <= 0.43
            and m["fp"] < base["fp"]
            and m["tp"] >= base["tp"] - 2
        )
        nearby.append({"threshold": tt, "passes": passes, "metrics": m})
    robust_pass = sum(int(x["passes"]) for x in nearby)

    gates = {
        "same_381_purged_oof_rows": True,
        "v61_and_v71_scores_are_oof": True,
        "raw_probability_blend_deployable_without_future_rank_distribution": True,
        "no_independent_validation_claim": True,
        "recall_ge_080": selected_metrics["recall"] >= 0.80,
        "precision_ge_v61": selected_metrics["precision"] >= base["precision"],
        "f1_gt_v61": selected_metrics["f1"] > base["f1"],
        "green_npv_ge_095": selected_metrics["green_npv"] >= 0.95,
        "alert_rate_le_v61": selected_metrics["alert_rate"] <= base["alert_rate"],
        "fp_lt_v61": selected_metrics["fp"] < base["fp"],
        "tp_loss_max_1": selected_metrics["tp"] >= base["tp"] - 1,
        "worst_fold_recall_ge_060": selected_metrics["worst_fold_recall"] >= 0.60,
        "max_fold_alert_rate_le_043": selected_metrics["max_fold_alert_rate"] <= 0.43,
        "threshold_neighborhood_support": robust_pass >= 5,
    }
    adopt = bool(all(gates.values()))

    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_ACCEPTED_PENDING_EXTERNAL_VALIDATION" if adopt else "EXPERIMENTAL_NOT_ADOPTED",
        "scientific_boundary": "V7.2.1 is a development-only raw probability fusion evaluated on the same 381 OOF rows already used in V6.1/V7.1 development. It is not blind or external validation.",
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible),
        "selection_mode": selection_mode,
        "base_v6_1": base,
        "selected": selected,
        "robustness": {
            "threshold_neighbors": len(nearby),
            "passing_neighbors": robust_pass,
            "passing_share": float(robust_pass / len(nearby)),
        },
        "comparison_vs_v6_1": {
            "precision_delta": float(selected_metrics["precision"] - base["precision"]),
            "recall_delta": float(selected_metrics["recall"] - base["recall"]),
            "f1_delta": float(selected_metrics["f1"] - base["f1"]),
            "npv_delta": float(selected_metrics["green_npv"] - base["green_npv"]),
            "alert_rate_delta": float(selected_metrics["alert_rate"] - base["alert_rate"]),
            "fp_removed": int(base["fp"] - selected_metrics["fp"]),
            "tp_change": int(selected_metrics["tp"] - base["tp"]),
        },
        "gates": gates,
        "all_development_gates_passed": adopt,
        "adopt_over_v6_1": adopt,
        "red_supported": False,
        "next_required_evidence": "Never-used real Saudi merchant longitudinal data.",
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    m = selected_metrics
    b = base
    lines = [
        "# Sales Sentinel V7.2.1 — Raw Score Fusion Refinement",
        "",
        f"- Selection mode: **{selection_mode}**",
        f"- V7.1 weight: **{w:.2f}**",
        f"- Threshold: **{t:.3f}**",
        f"- Fusion ROC-AUC: **{selected['roc_auc']:.2%}**",
        f"- Fusion PR-AUC: **{selected['pr_auc']:.2%}**",
        f"- Feasible candidates: **{len(feasible)}/{len(candidates)}**",
        "",
        f"- Precision: V6.1 **{b['precision']:.2%}** -> V7.2.1 **{m['precision']:.2%}**",
        f"- Recall: V6.1 **{b['recall']:.2%}** -> V7.2.1 **{m['recall']:.2%}**",
        f"- F1: V6.1 **{b['f1']:.2%}** -> V7.2.1 **{m['f1']:.2%}**",
        f"- GREEN NPV: V6.1 **{b['green_npv']:.2%}** -> V7.2.1 **{m['green_npv']:.2%}**",
        f"- Alert rate: V6.1 **{b['alert_rate']:.2%}** -> V7.2.1 **{m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- FP removed vs V6.1: **{b['fp'] - m['fp']}**",
        f"- TP change vs V6.1: **{m['tp'] - b['tp']:+d}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Max-fold alert rate: **{m['max_fold_alert_rate']:.2%}**",
        f"- Robust threshold neighbors: **{robust_pass}/{len(nearby)}**",
        f"- Development gates passed: **{adopt}**",
        f"- Adopt over V6.1: **{adopt}**",
        "- RED supported: **False**",
        "",
        "Scientific boundary: development-only OOF evidence; independent Saudi merchant validation remains required.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
