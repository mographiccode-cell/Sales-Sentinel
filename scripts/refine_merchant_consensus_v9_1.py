from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import train_merchant_error_corrector_v7_5 as v75

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V9.1-CONTINUOUS-RANK-CONSENSUS"
V9_OOF = ROOT / "reports" / "merchant_continuous_ratio_v9" / "oof_selected_predictions.csv"
V76_OOF = ROOT / "reports" / "merchant_ensemble_v7_6" / "oof_ensemble_predictions.csv"
V61_DIAG = ROOT / "reports" / "merchant_market_fusion_v6_1" / "oof_policy_diagnostics.csv"
OUT = ROOT / "reports" / "merchant_consensus_v9_1"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_predictions.csv"


def apply_rule(base, strong, v9risk, v76, ratio, rule):
    scope, q9, q76, ratio_min = rule
    pred = np.asarray(base, bool).copy()
    if scope == "none":
        return pred
    mask = (~strong) if scope == "nonstrong" else np.ones(len(pred), bool)
    veto = pred & mask & (v9risk < q9) & (v76 < q76) & (ratio > ratio_min)
    pred[veto] = False
    return pred


def rules():
    out = [("none", 0.0, 0.0, .90)]
    for scope, q9, q76, r in product(
        ["nonstrong", "any"],
        [.15, .20, .25, .30, .35, .40, .45, .50, .55, .60],
        [.20, .25, .30, .35, .40, .45, .50, .55, .60],
        [.90, .95, 1.00, 1.05],
    ):
        out.append((scope, q9, q76, r))
    return out


def prequential(y, folds, base, strong, v9risk, v76, ratio):
    final = np.asarray(base, bool).copy(); details = []
    for f in sorted(np.unique(folds)):
        cur = folds == f
        if f == 0:
            details.append({"fold_id": int(f), "mode": "v9_bootstrap"})
            continue
        hist = folds < f
        bm = v75.metrics(y[hist], base[hist], folds[hist])
        best = None
        for rule in rules():
            p = apply_rule(base[hist], strong[hist], v9risk[hist], v76[hist], ratio[hist], rule)
            m = v75.metrics(y[hist], p, folds[hist])
            feasible = (
                m["recall"] >= bm["recall"] and
                m["green_npv"] >= bm["green_npv"] - .001 and
                m["fp"] <= bm["fp"] and
                m["f1"] >= bm["f1"] and
                m["alert_rate"] <= bm["alert_rate"]
            )
            key = (int(feasible), m["f1"], m["precision"], -m["fp"], m["recall"], -m["alert_rate"])
            if best is None or key > best[0]:
                best = (key, rule, m, feasible)
        _, rule, hm, feasible = best
        if not feasible:
            rule = ("none", 0.0, 0.0, .90)
        final[cur] = apply_rule(base[cur], strong[cur], v9risk[cur], v76[cur], ratio[cur], rule)
        details.append({"fold_id": int(f), "history_rule": list(rule), "history_feasible": bool(feasible), "history_metrics": hm})
    return final, details


def main():
    a = pd.read_csv(V9_OOF).sort_values(["fold_id", "date"]).reset_index(drop=True)
    b = pd.read_csv(V76_OOF).sort_values(["fold_id", "date"]).reset_index(drop=True)
    diag = pd.read_csv(V61_DIAG)
    if len(a) != len(b) or not np.array_equal(a.y.to_numpy(int), b.y.to_numpy(int)) or not np.array_equal(a.fold_id.to_numpy(int), b.fold_id.to_numpy(int)):
        raise RuntimeError("V9/V7.6 alignment mismatch")
    y = a.y.to_numpy(int); folds = a.fold_id.to_numpy(int)
    base = a.v9_pred.to_numpy(bool)
    v9risk = a.v9_risk.to_numpy(float)
    ratio = a.pred_future_ratio.to_numpy(float)
    v76 = b.ensemble_score.to_numpy(float)
    strong, _, _, _ = v75.base_components(diag)
    bm = v75.metrics(y, base, folds)
    pred, details = prequential(y, folds, base, strong, v9risk, v76, ratio)
    m = v75.metrics(y, pred, folds)
    strict = bool(
        m["recall"] >= bm["recall"] and
        m["precision"] > bm["precision"] and
        m["f1"] > bm["f1"] and
        m["green_npv"] >= bm["green_npv"] and
        m["alert_rate"] <= bm["alert_rate"] and
        m["fp"] < bm["fp"] and
        m["worst_fold_recall"] >= bm["worst_fold_recall"]
    )

    # Diagnostic full-development oracle only; not validation evidence.
    oracle = None
    for rule in rules():
        p = apply_rule(base, strong, v9risk, v76, ratio, rule)
        om = v75.metrics(y, p, folds)
        feasible = om["recall"] >= bm["recall"] and om["green_npv"] >= bm["green_npv"] and om["fp"] < bm["fp"] and om["f1"] > bm["f1"] and om["worst_fold_recall"] >= bm["worst_fold_recall"]
        key = (int(feasible), om["f1"], om["precision"], -om["fp"], om["recall"])
        if oracle is None or key > oracle[0]: oracle = (key, rule, om, feasible)

    pd.DataFrame({
        "date": a.date, "y": y, "fold_id": folds, "v9_pred": base.astype(int),
        "v9_risk": v9risk, "v76_score": v76, "pred_future_ratio": ratio,
        "v9_1_pred": pred.astype(int),
    }).to_csv(OOF, index=False)

    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_BEST" if strict else "EXPERIMENTAL_V9_REMAINS_BEST",
        "scientific_boundary": "V9.1 selects consensus-veto rules prequentially from earlier folds only. The full-development oracle is diagnostic only. External Saudi merchant validation remains required.",
        "rule_count": len(rules()),
        "v9": bm,
        "v9_1": m,
        "prequential_details": details,
        "strictly_dominates_v9": strict,
        "development_oracle": {"rule": list(oracle[1]), "metrics": oracle[2], "strict_feasible": bool(oracle[3])},
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Sales Sentinel V9.1 — Continuous + Rank Consensus Verifier", "",
        f"- Status: **{report['status']}**",
        f"- Rules evaluated causally: **{len(rules())}**", "",
        f"- Precision: V9 **{bm['precision']:.2%}** -> V9.1 **{m['precision']:.2%}**",
        f"- Recall: V9 **{bm['recall']:.2%}** -> V9.1 **{m['recall']:.2%}**",
        f"- F1: V9 **{bm['f1']:.2%}** -> V9.1 **{m['f1']:.2%}**",
        f"- NPV: V9 **{bm['green_npv']:.2%}** -> V9.1 **{m['green_npv']:.2%}**",
        f"- Alert rate: V9 **{bm['alert_rate']:.2%}** -> V9.1 **{m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Strictly dominates V9: **{strict}**", "",
        f"- Development oracle rule: **{list(oracle[1])}**",
        f"- Development oracle TP/FP/FN/TN: **{oracle[2]['tp']}/{oracle[2]['fp']}/{oracle[2]['fn']}/{oracle[2]['tn']}**", "",
        "Scientific boundary: development evidence only; external Saudi merchant validation remains pending.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
