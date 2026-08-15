from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import train_merchant_error_corrector_v7_5 as v75
import train_merchant_saudi_calendar_v13 as v13
import refine_merchant_calendar_safeguard_v13_1 as v131

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V13.1-FROZEN-DEVELOPMENT-CANDIDATE"
V11 = ROOT / "reports" / "merchant_multihorizon_v11" / "oof_predictions.csv"
V131 = ROOT / "reports" / "merchant_calendar_safeguard_v13_1" / "oof_predictions.csv"
V131_REPORT = ROOT / "reports" / "merchant_calendar_safeguard_v13_1" / "development_report.json"
PANEL = ROOT / "data" / "merchant_v7_1" / "merchant_feature_panel_v7_1.csv"
OUT = ROOT / "reports" / "merchant_v13_1_final"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "finalization_report.json"
SUMMARY = OUT / "finalization_summary.md"
CHANGED = OUT / "changed_alerts.csv"
NEIGHBORS = OUT / "neighbor_robustness.csv"


def candidate_with_feature_mode(b, x, strong, market, cfg, include_calendar: bool):
    y = b.y.to_numpy(int); folds = b.fold_id.to_numpy(int); base = b.v11_pred.to_numpy(bool)
    core = [c for c in x.columns if not c.startswith(("cal__", "calx__"))]
    cal = [c for c in x.columns if c.startswith(("cal__", "calx__"))]
    cols = core + cal if include_calendar else core
    final = base.copy(); details = []
    risk_mean = x["risk_mean"].to_numpy(float)
    risk_max = x["risk_max"].to_numpy(float)
    risk_min = x["risk_min"].to_numpy(float)
    for f in sorted(np.unique(folds)):
        cur = folds == f
        if f == 0:
            details.append({"fold_id": int(f), "mode": "v11_bootstrap"}); continue
        hist = folds < f; ha = hist & base
        ptr, pcur = v13.fit_predict(x.loc[ha, cols], y[ha], x.loc[cur, cols], cfg["C"])
        tp = ptr[y[ha] == 1]
        thr = float(max(0.0, np.quantile(tp, cfg["tp_quantile"]) - cfg["margin"]))
        if cfg["scope"] == "market":
            scope = market[cur] & (~strong[cur])
        elif cfg["scope"] == "nonstrong":
            scope = ~strong[cur]
        else:
            scope = np.ones(cur.sum(), bool)
        weak = (
            (risk_mean[cur] < cfg["mean_guard"]) &
            (risk_max[cur] < cfg["max_guard"]) &
            (risk_min[cur] < cfg["min_guard"])
        )
        cur_base = base[cur].copy()
        veto = cur_base & scope & (pcur < thr) & weak
        final[cur] = cur_base & (~veto)
        details.append({"fold_id": int(f), "threshold": thr, "vetoes": int(veto.sum())})
    return final, details


def passes_contract(m, base):
    return bool(
        m["recall"] >= base["recall"] and
        m["green_npv"] >= base["green_npv"] and
        m["precision"] > base["precision"] and
        m["f1"] > base["f1"] and
        m["fp"] < base["fp"] and
        m["worst_fold_recall"] >= base["worst_fold_recall"] and
        m["alert_rate"] <= base["alert_rate"]
    )


def main():
    rep = json.loads(V131_REPORT.read_text(encoding="utf-8"))
    cfg = rep["selected"]["config"]
    b, _, _, x, strong, market = v13.build_features()
    y = b.y.to_numpy(int); folds = b.fold_id.to_numpy(int); v11_pred = b.v11_pred.to_numpy(bool)
    base_metrics = v75.metrics(y, v11_pred, folds)

    saved = pd.read_csv(V131).sort_values(["fold_id", "date"]).reset_index(drop=True)
    if len(saved) != len(b) or not np.array_equal(saved.y.to_numpy(int), y):
        raise RuntimeError("V13.1 saved OOF mismatch")
    selected_pred = saved.v13_1_pred.to_numpy(bool)
    selected_metrics = v75.metrics(y, selected_pred, folds)
    if (selected_metrics["tp"], selected_metrics["fp"], selected_metrics["fn"], selected_metrics["tn"]) != (52,65,11,253):
        raise RuntimeError("Frozen V13.1 reconstruction mismatch")

    # Feature ablation using the exact same causal procedure and selected hyperparameters.
    cal_pred, cal_details = candidate_with_feature_mode(b, x, strong, market, cfg, True)
    core_pred, core_details = candidate_with_feature_mode(b, x, strong, market, cfg, False)
    cal_metrics = v75.metrics(y, cal_pred, folds)
    core_metrics = v75.metrics(y, core_pred, folds)
    if not np.array_equal(cal_pred, selected_pred):
        raise RuntimeError("Calendar reconstruction does not match persisted V13.1 predictions")

    # Inspect exactly which alerts changed from V11.
    panel = pd.read_csv(PANEL, parse_dates=["date"])[["date", "future_ratio"]]
    changed = pd.DataFrame({
        "date": pd.to_datetime(b.date), "y": y, "fold_id": folds,
        "v11_pred": v11_pred.astype(int), "v13_1_pred": selected_pred.astype(int),
        "calendar_special": x["cal__special_regime"].to_numpy(float),
        "is_ramadan": x["cal__is_ramadan"].to_numpy(float),
        "pre_fitr_7": x["cal__pre_fitr_7"].to_numpy(float),
        "fitr_day0_3": x["cal__fitr_day0_3"].to_numpy(float),
        "pre_adha_7": x["cal__pre_adha_7"].to_numpy(float),
        "adha_day0_3": x["cal__adha_day0_3"].to_numpy(float),
        "risk_mean": x["risk_mean"].to_numpy(float), "risk_max": x["risk_max"].to_numpy(float),
        "risk_min": x["risk_min"].to_numpy(float),
    })
    changed = changed[changed.v11_pred != changed.v13_1_pred].merge(panel, on="date", how="left")
    changed.to_csv(CHANGED, index=False)

    # Pre-specified local neighborhood: robustness audit, not model selection.
    vals = {
        "C": sorted(set([.20, cfg["C"], .40])),
        "margin": sorted(set([0.0, cfg["margin"], .010])),
        "mean_guard": sorted(set([.40, cfg["mean_guard"], .50])),
        "max_guard": sorted(set([.80, cfg["max_guard"], .90])),
        "min_guard": sorted(set([.20, cfg["min_guard"], .30])),
    }
    rows = []
    for C, margin, mg, xg, ng in product(vals["C"], vals["margin"], vals["mean_guard"], vals["max_guard"], vals["min_guard"]):
        c = dict(cfg); c.update({"C":C,"margin":margin,"mean_guard":mg,"max_guard":xg,"min_guard":ng})
        p, _ = candidate_with_feature_mode(b, x, strong, market, c, True)
        m = v75.metrics(y, p, folds)
        rows.append({
            "C":C,"margin":margin,"mean_guard":mg,"max_guard":xg,"min_guard":ng,
            "precision":m["precision"],"recall":m["recall"],"f1":m["f1"],
            "npv":m["green_npv"],"alert_rate":m["alert_rate"],"tp":m["tp"],"fp":m["fp"],
            "fn":m["fn"],"tn":m["tn"],"worst_fold_recall":m["worst_fold_recall"],
            "passes_v11_contract":passes_contract(m,base_metrics),
            "preserves_selected_or_better":bool(m["tp"]>=52 and m["fp"]<=65 and m["fn"]<=11),
        })
    nr = pd.DataFrame(rows); nr.to_csv(NEIGHBORS, index=False)

    fold_delta = []
    for f in sorted(np.unique(folds)):
        a = base_metrics["per_fold"][f]; z = selected_metrics["per_fold"][f]
        fold_delta.append({
            "fold_id": int(f), "fp_before": int(a["fp"]), "fp_after": int(z["fp"]),
            "fp_reduction": int(a["fp"]-z["fp"]), "tp_before": int(a["tp"]), "tp_after": int(z["tp"]),
            "recall_before": a["recall"], "recall_after": z["recall"],
            "precision_before": a["precision"], "precision_after": z["precision"],
        })

    improved_folds = sum(r["fp_reduction"] > 0 for r in fold_delta)
    degraded_tp_folds = sum(r["tp_after"] < r["tp_before"] for r in fold_delta)
    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_FROZEN_PENDING_FRESH_EXTERNAL_VALIDATION",
        "scientific_boundary": "V13.1 is the best development OOF decision policy found, but its incremental gain over V11 is concentrated in one outer fold and follows extensive reuse of the same 381 OOF rows. The frozen metrics are development evidence, not independent validation. No additional hyperparameter search on these OOF rows should be used to claim further validation gains.",
        "selected_config": cfg,
        "v11_reference": base_metrics,
        "v13_1_primary_metrics": selected_metrics,
        "calendar_ablation": {
            "with_calendar": cal_metrics,
            "same_config_without_calendar": core_metrics,
            "calendar_fp_delta": int(core_metrics["fp"]-cal_metrics["fp"]),
            "calendar_tp_delta": int(cal_metrics["tp"]-core_metrics["tp"]),
        },
        "fold_delta": fold_delta,
        "fold_robustness": {
            "folds_with_fp_improvement": int(improved_folds),
            "folds_with_tp_degradation": int(degraded_tp_folds),
            "total_folds": int(len(fold_delta)),
        },
        "changed_alerts": {
            "count": int(len(changed)),
            "all_removed_are_true_negatives": bool(len(changed)>0 and (changed.y==0).all() and (changed.v11_pred==1).all() and (changed.v13_1_pred==0).all()),
            "calendar_special_share": float(changed.calendar_special.mean()) if len(changed) else None,
        },
        "neighbor_robustness": {
            "tested": int(len(nr)),
            "pass_v11_contract": int(nr.passes_v11_contract.sum()),
            "pass_v11_contract_rate": float(nr.passes_v11_contract.mean()),
            "selected_or_better": int(nr.preserves_selected_or_better.sum()),
            "selected_or_better_rate": float(nr.preserves_selected_or_better.mean()),
            "median_fp": float(nr.fp.median()),
            "min_fp_with_tp52": int(nr.loc[nr.tp>=52,"fp"].min()) if (nr.tp>=52).any() else None,
        },
        "external_validation_pending": True,
        "red_supported": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    m=selected_metrics; ab=report["calendar_ablation"]; rob=report["neighbor_robustness"]
    lines = [
        "# Sales Sentinel V13.1 — Frozen Development Candidate", "",
        f"- Status: **{report['status']}**",
        f"- Precision / Recall / F1: **{m['precision']:.2%} / {m['recall']:.2%} / {m['f1']:.2%}**",
        f"- Accuracy / Balanced Accuracy: **{m['accuracy']:.2%} / {m['balanced_accuracy']:.2%}**",
        f"- GREEN NPV / Alert rate: **{m['green_npv']:.2%} / {m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**", "",
        "## Incremental robustness", 
        f"- Folds with FP improvement vs V11: **{improved_folds}/{len(fold_delta)}**",
        f"- Folds with TP degradation: **{degraded_tp_folds}/{len(fold_delta)}**",
        f"- Neighbor configs passing V11 contract: **{rob['pass_v11_contract']}/{rob['tested']} ({rob['pass_v11_contract_rate']:.1%})**",
        f"- Neighbor configs matching/exceeding TP52 + FP<=65: **{rob['selected_or_better']}/{rob['tested']} ({rob['selected_or_better_rate']:.1%})**",
        f"- Same selected config without calendar TP/FP: **{ab['same_config_without_calendar']['tp']}/{ab['same_config_without_calendar']['fp']}**",
        f"- With calendar TP/FP: **{ab['with_calendar']['tp']}/{ab['with_calendar']['fp']}**",
        f"- Alerts changed vs V11: **{len(changed)}**, all true-negative removals: **{report['changed_alerts']['all_removed_are_true_negatives']}**", "",
        "Important: the incremental V13.1 gain is concentrated in one fold. Freeze this version and require fresh longitudinal Saudi merchant data or a new untouched future period before claiming further generalization.",
    ]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
