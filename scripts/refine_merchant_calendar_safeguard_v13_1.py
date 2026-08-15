from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import train_merchant_error_corrector_v7_5 as v75
import train_merchant_saudi_calendar_v13 as v13

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V13.1-CALENDAR-SAFEGUARDED-VERIFIER"
BASE = ROOT / "reports" / "merchant_multihorizon_v11" / "oof_predictions.csv"
OUT = ROOT / "reports" / "merchant_calendar_safeguard_v13_1"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_predictions.csv"


def candidate(b, x, strong, market, cfg):
    y = b.y.to_numpy(int); folds = b.fold_id.to_numpy(int); base = b.v11_pred.to_numpy(bool)
    core = [c for c in x.columns if not c.startswith(("cal__", "calx__"))]
    cal = [c for c in x.columns if c.startswith(("cal__", "calx__"))]
    cols = core + cal
    final = base.copy(); details = []
    risk_mean = x["risk_mean"].to_numpy(float)
    risk_max = x["risk_max"].to_numpy(float)
    risk_min = x["risk_min"].to_numpy(float)

    for f in sorted(np.unique(folds)):
        cur = folds == f
        if f == 0:
            details.append({"fold_id": int(f), "mode": "v11_bootstrap"}); continue
        hist = folds < f; ha = hist & base
        if ha.sum() < 20 or len(np.unique(y[ha])) < 2:
            details.append({"fold_id": int(f), "mode": "insufficient_history"}); continue
        ptr, pcur = v13.fit_predict(x.loc[ha, cols], y[ha], x.loc[cur, cols], cfg["C"])
        tp = ptr[y[ha] == 1]
        thr = float(max(0.0, np.quantile(tp, cfg["tp_quantile"]) - cfg["margin"]))
        if cfg["scope"] == "market":
            scope = market[cur] & (~strong[cur])
        elif cfg["scope"] == "nonstrong":
            scope = ~strong[cur]
        else:
            scope = np.ones(cur.sum(), bool)

        cur_base = base[cur].copy()
        # Safeguard: a veto is allowed only when ALL risk views are weak enough.
        weak_consensus = (
            (risk_mean[cur] < cfg["mean_guard"]) &
            (risk_max[cur] < cfg["max_guard"]) &
            (risk_min[cur] < cfg["min_guard"])
        )
        veto = cur_base & scope & (pcur < thr) & weak_consensus
        final[cur] = cur_base & (~veto)
        details.append({
            "fold_id": int(f), "history_alerts": int(ha.sum()), "history_tp": int(y[ha].sum()),
            "threshold": thr, "vetoes": int(veto.sum()), "mode": "verified",
        })
    return final, details


def main():
    b, _, _, x, strong, market = v13.build_features()
    y = b.y.to_numpy(int); folds = b.fold_id.to_numpy(int); base = b.v11_pred.to_numpy(bool)
    bm = v75.metrics(y, base, folds)
    configs = []
    for C, margin, scope, mean_guard, max_guard, min_guard in product(
        [.10, .30, .60], [.005, .01, .02, .03], ["market", "nonstrong", "any"],
        [.45, .50, .55, .60, .65], [.55, .65, .75, .85], [.25, .35, .45, .55]
    ):
        configs.append({
            "C": C, "tp_quantile": 0.0, "margin": margin, "scope": scope,
            "mean_guard": mean_guard, "max_guard": max_guard, "min_guard": min_guard,
        })

    rows, preds = [], []
    for i, cfg in enumerate(configs):
        p, details = candidate(b, x, strong, market, cfg)
        m = v75.metrics(y, p, folds)
        adopt = bool(
            m["recall"] >= bm["recall"] and m["green_npv"] >= bm["green_npv"] and
            m["precision"] > bm["precision"] and m["f1"] > bm["f1"] and
            m["fp"] < bm["fp"] and m["worst_fold_recall"] >= bm["worst_fold_recall"] and
            m["alert_rate"] <= bm["alert_rate"]
        )
        rows.append({"config_id": i, "config": cfg, "metrics": m, "strictly_dominates_v11": adopt, "details": details})
        preds.append(p)

    def key(r):
        m = r["metrics"]
        return (int(r["strictly_dominates_v11"]), m["f1"], m["precision"], -m["fp"], m["recall"], m["green_npv"], -m["alert_rate"])

    sel = max(rows, key=key); pred = preds[sel["config_id"]]; m = sel["metrics"]
    pd.DataFrame({
        "date": b.date, "y": y, "fold_id": folds, "v11_pred": base.astype(int),
        "risk_mean": x.risk_mean, "risk_max": x.risk_max, "risk_min": x.risk_min,
        "calendar_special": x["cal__special_regime"], "v13_1_pred": pred.astype(int),
    }).to_csv(OOF, index=False)

    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_BEST" if sel["strictly_dominates_v11"] else "EXPERIMENTAL_V13_CORE_REMAINS_BEST",
        "scientific_boundary": "V13.1 is a single targeted refinement motivated by the observed V13 calendar trade-off. All verifier fits remain earlier-fold-only, but the configuration grid is selected on previously reused development folds; no further development tuning should be treated as independent validation.",
        "candidate_count": len(rows), "v11": bm, "selected": sel,
        "top_candidates": sorted(rows, key=key, reverse=True)[:10], "red_supported": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Sales Sentinel V13.1 — Calendar Safeguarded Verifier", "",
        f"- Status: **{report['status']}**", f"- Candidates: **{len(rows)}**",
        f"- Selected: **{sel['config']}**", "",
        f"- Precision: V11 **{bm['precision']:.2%}** -> V13.1 **{m['precision']:.2%}**",
        f"- Recall: V11 **{bm['recall']:.2%}** -> V13.1 **{m['recall']:.2%}**",
        f"- F1: V11 **{bm['f1']:.2%}** -> V13.1 **{m['f1']:.2%}**",
        f"- NPV: V11 **{bm['green_npv']:.2%}** -> V13.1 **{m['green_npv']:.2%}**",
        f"- Alert rate: V11 **{bm['alert_rate']:.2%}** -> V13.1 **{m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Strictly dominates V11: **{sel['strictly_dominates_v11']}**", "",
        "Scientific boundary: final targeted development refinement; fresh external validation is required before any production claim.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
