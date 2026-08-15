from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import train_merchant_error_corrector_v7_5 as v75
import train_merchant_target_refinement_v8 as v8

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V9.2-PREQUENTIAL-META-ALERT-VERIFIER"
V9 = ROOT / "reports" / "merchant_continuous_ratio_v9" / "oof_selected_predictions.csv"
V76 = ROOT / "reports" / "merchant_ensemble_v7_6" / "oof_ensemble_predictions.csv"
V82 = ROOT / "reports" / "merchant_alert_verifier_v8_2" / "oof_predictions.csv"
DIAG = ROOT / "reports" / "merchant_market_fusion_v6_1" / "oof_policy_diagnostics.csv"
PANEL = ROOT / "data" / "merchant_v7_1" / "merchant_feature_panel_v7_1.csv"
OUT = ROOT / "reports" / "merchant_meta_verifier_v9_2"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_predictions.csv"


def build_features():
    a = pd.read_csv(V9).sort_values(["fold_id", "date"]).reset_index(drop=True)
    b = pd.read_csv(V76).sort_values(["fold_id", "date"]).reset_index(drop=True)
    c = pd.read_csv(V82).sort_values(["fold_id", "date"]).reset_index(drop=True)
    d = pd.read_csv(DIAG).reset_index(drop=True)
    p = pd.read_csv(PANEL, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    p = v8.add_hard_negative_features(p)
    pp = []
    for fid, _, va in v75.windows(p):
        q = p.loc[va].copy(); q["fold_id"] = fid; pp.append(q)
    po = pd.concat(pp, ignore_index=True).sort_values(["fold_id", "date"]).reset_index(drop=True)
    if not (len(a) == len(b) == len(c) == len(d) == len(po) == 381):
        raise RuntimeError("length mismatch")
    y = a.y.to_numpy(int)
    if not np.array_equal(y, b.y.to_numpy(int)) or not np.array_equal(y, c.y.to_numpy(int)):
        raise RuntimeError("target mismatch")
    x = pd.DataFrame(index=a.index)
    x["v9_risk"] = a.v9_risk.astype(float)
    x["v9_pred_ratio"] = a.pred_future_ratio.astype(float)
    x["v76_score"] = b.ensemble_score.astype(float)
    x["v8_rank"] = c.v8_rank.astype(float)
    x["risk_consensus"] = (x.v9_risk + x.v76_score + x.v8_rank) / 3.0
    x["v9_v76_gap"] = x.v9_risk - x.v76_score
    x["ratio_below_1"] = 1.0 - x.v9_pred_ratio
    for col in [
        "merchant_logreg", "merchant_extra", "merchant_mean", "merchant_disagreement",
        "market_v3__risk_mean", "market_v3__risk_max", "market_v3__risk_p90",
        "market_v3__risk_share_25", "market_v3__precursor_mean",
    ]:
        if col in d.columns:
            x[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    strong, quiet, market, _ = v75.base_components(d)
    x["branch_strong"] = strong.astype(float)
    x["branch_quiet"] = quiet.astype(float)
    x["branch_market"] = market.astype(float)
    for col in [cc for cc in po.columns if cc.startswith("v8__")]:
        x[col] = pd.to_numeric(po[col], errors="coerce").fillna(0.0).to_numpy()
    return a, b, d, x, strong, quiet, market


def fit_predict(Xtr, ytr, Xva, C):
    sc = StandardScaler()
    A = sc.fit_transform(Xtr); B = sc.transform(Xva)
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=42)
    m.fit(A, ytr)
    return m.predict_proba(A)[:, 1], m.predict_proba(B)[:, 1]


def candidate(a, b, x, strong, market, cfg):
    y = a.y.to_numpy(int); folds = a.fold_id.to_numpy(int); base = a.v9_pred.to_numpy(bool)
    v9risk = a.v9_risk.to_numpy(float); v76 = b.ensemble_score.to_numpy(float)
    final = base.copy(); details = []
    core = [cc for cc in x.columns if not cc.startswith("v8__")]
    hard = [cc for cc in x.columns if cc.startswith("v8__")]
    cols = core if cfg["features"] == "core" else core + hard
    for f in sorted(np.unique(folds)):
        cur = folds == f
        if f == 0:
            details.append({"fold_id": int(f), "mode": "v9_bootstrap"}); continue
        hist = folds < f; ha = hist & base
        if ha.sum() < 20 or len(np.unique(y[ha])) < 2:
            details.append({"fold_id": int(f), "mode": "insufficient_history"}); continue
        ptr, pcur = fit_predict(x.loc[ha, cols], y[ha], x.loc[cur, cols], cfg["C"])
        tp_probs = ptr[y[ha] == 1]
        if len(tp_probs) == 0:
            details.append({"fold_id": int(f), "mode": "no_historical_tp"}); continue
        thr = float(max(0.0, np.quantile(tp_probs, cfg["tp_quantile"]) - cfg["margin"]))
        if cfg["scope"] == "market":
            scope = market[cur] & (~strong[cur])
        elif cfg["scope"] == "nonstrong":
            scope = ~strong[cur]
        else:
            scope = np.ones(cur.sum(), bool)
        cur_base = base[cur].copy()
        veto = cur_base & scope & (pcur < thr) & (v9risk[cur] < cfg["v9_guard"]) & (v76[cur] < cfg["v76_guard"])
        cur_pred = cur_base & (~veto)
        final[cur] = cur_pred
        details.append({
            "fold_id": int(f), "history_alerts": int(ha.sum()), "history_tp": int(y[ha].sum()),
            "threshold": thr, "vetoes": int(veto.sum()), "mode": "verified",
        })
    return final, details


def main():
    a, b, d, x, strong, quiet, market = build_features()
    y = a.y.to_numpy(int); folds = a.fold_id.to_numpy(int); base = a.v9_pred.to_numpy(bool)
    bm = v75.metrics(y, base, folds)
    configs = []
    for features, C, tpq, margin, scope, g9, g76 in product(
        ["core", "hard"], [.05, .10, .50], [0.0, .10, .20], [.01, .03],
        ["market", "nonstrong", "any"], [.35, .45, .55], [.40, .50, .60],
    ):
        configs.append({
            "features": features, "C": C, "tp_quantile": tpq, "margin": margin,
            "scope": scope, "v9_guard": g9, "v76_guard": g76,
        })
    rows, preds = [], []
    for i, cfg in enumerate(configs):
        p, details = candidate(a, b, x, strong, market, cfg)
        m = v75.metrics(y, p, folds)
        adopt = bool(
            m["recall"] >= bm["recall"] and m["green_npv"] >= bm["green_npv"] and
            m["precision"] > bm["precision"] and m["f1"] > bm["f1"] and
            m["fp"] < bm["fp"] and m["worst_fold_recall"] >= bm["worst_fold_recall"] and
            m["alert_rate"] <= bm["alert_rate"]
        )
        rows.append({"config_id": i, "config": cfg, "metrics": m, "strictly_dominates_v9": adopt, "details": details})
        preds.append(p)
    def key(r):
        m = r["metrics"]
        return (int(r["strictly_dominates_v9"]), m["f1"], m["precision"], -m["fp"], m["recall"], m["green_npv"], -m["alert_rate"])
    sel = max(rows, key=key); pred = preds[sel["config_id"]]; m = sel["metrics"]
    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_BEST" if sel["strictly_dominates_v9"] else "EXPERIMENTAL_V9_REMAINS_BEST",
        "scientific_boundary": "Each V9.2 fold uses a verifier fitted only to earlier V9 alerts. Hyperparameter selection remains development selection on previously used folds; external Saudi merchant validation is required.",
        "candidate_count": len(rows), "v9": bm, "selected": sel,
        "top_candidates": sorted(rows, key=key, reverse=True)[:10], "red_supported": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({
        "date": a.date, "y": y, "fold_id": folds, "v9_pred": base.astype(int),
        "v9_risk": a.v9_risk, "v76_score": b.ensemble_score, "v9_2_pred": pred.astype(int),
    }).to_csv(OOF, index=False)
    lines = [
        "# Sales Sentinel V9.2 — Prequential Meta Alert Verifier", "",
        f"- Status: **{report['status']}**", f"- Candidates: **{len(rows)}**",
        f"- Selected: **{sel['config']}**", "",
        f"- Precision: V9 **{bm['precision']:.2%}** -> V9.2 **{m['precision']:.2%}**",
        f"- Recall: V9 **{bm['recall']:.2%}** -> V9.2 **{m['recall']:.2%}**",
        f"- F1: V9 **{bm['f1']:.2%}** -> V9.2 **{m['f1']:.2%}**",
        f"- NPV: V9 **{bm['green_npv']:.2%}** -> V9.2 **{m['green_npv']:.2%}**",
        f"- Alert rate: V9 **{bm['alert_rate']:.2%}** -> V9.2 **{m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Strictly dominates V9: **{sel['strictly_dominates_v9']}**", "",
        "Scientific boundary: development-selected causal verifier; external validation remains pending.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
