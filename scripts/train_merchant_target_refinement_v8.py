from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

import train_merchant_error_corrector_v7_5 as v75

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V8-TARGET-REFINEMENT-HARD-NEGATIVE"
SEED = 42
PANEL = ROOT / "data" / "merchant_v7_1" / "merchant_feature_panel_v7_1.csv"
V61_DIAG = ROOT / "reports" / "merchant_market_fusion_v6_1" / "oof_policy_diagnostics.csv"
V75_FUSED = ROOT / "reports" / "merchant_error_corrector_v7_5" / "oof_fused_predictions.csv"
OUT = ROOT / "reports" / "merchant_target_refinement_v8"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_candidate_scores.csv"
FUSED = OUT / "oof_fused_predictions.csv"
AUDIT = OUT / "v7_5_error_audit.json"
META = {"date", "future_ratio", "future7_sales", "baseline28_daily", "target"}


def num(df, col):
    if col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_hard_negative_features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    # Short-vs-long sales movement: transient shocks often recover while true declines have breadth/persistence.
    s1 = num(x, "merchant__net_sales_sar__change_1")
    s7 = num(x, "merchant__net_sales_sar__change_7")
    s14 = num(x, "merchant__net_sales_sar__change_14")
    s28 = num(x, "merchant__net_sales_sar__change_28")
    r7 = num(x, "merchant__net_sales_sar__ratio_mean_7")
    r14 = num(x, "merchant__net_sales_sar__ratio_mean_14")
    r28 = num(x, "merchant__net_sales_sar__ratio_mean_28")
    z7 = num(x, "merchant__net_sales_sar__z_7")
    z28 = num(x, "merchant__net_sales_sar__z_28")
    market7 = num(x, "market__sama_weekly_market_index__change_7")
    inv7 = num(x, "merchant__invoice_count__change_7")
    cust7 = num(x, "merchant__unique_observed_customers__change_7")
    prod7 = num(x, "merchant__unique_products__change_7")
    ret7 = num(x, "merchant__return_rate_value__change_7")
    breadth = num(x, "catregime__share_negative_sales_change7")
    severe_breadth = num(x, "catregime__share_severe_sales_change7")
    inv_breadth = num(x, "catregime__share_negative_invoice_change7")
    cust_breadth = num(x, "catregime__share_negative_customer_change7")
    below90 = num(x, "catregime__share_sales_below_090_ma7")
    above110 = num(x, "catregime__share_sales_above_110_ma7")

    x["v8__sales_short_long_change_gap"] = s7 - s28
    x["v8__sales_1d_vs_7d_gap"] = s1 - s7
    x["v8__sales_7d_vs_14d_gap"] = s7 - s14
    x["v8__sales_ratio_7_28_gap"] = r7 - r28
    x["v8__sales_ratio_7_14_gap"] = r7 - r14
    x["v8__sales_z_7_28_gap"] = z7 - z28
    x["v8__merchant_vs_market_7d"] = s7 - market7
    x["v8__sales_vs_invoice_7d"] = s7 - inv7
    x["v8__sales_vs_customer_7d"] = s7 - cust7
    x["v8__sales_vs_product_7d"] = s7 - prod7
    x["v8__commercial_breadth_mean"] = (breadth + inv_breadth + cust_breadth) / 3.0
    x["v8__decline_breadth_strength"] = (breadth + severe_breadth + below90) / 3.0
    x["v8__breadth_minus_sales_shock"] = breadth - np.clip(-s7, 0, None)
    x["v8__rebound_signature"] = np.clip(-s1, 0, None) * np.clip(s7 - s1, 0, None)
    x["v8__persistent_decline_signature"] = np.clip(-s7, 0, None) * np.clip(-s14, 0, None) * (0.5 + breadth)
    x["v8__temporary_category_shock"] = np.clip(-s7, 0, None) * np.clip(above110 + (1.0 - breadth), 0, 2)
    x["v8__return_distortion"] = np.abs(ret7) * np.clip(-s7, 0, None)
    x["v8__trend_consensus"] = (np.sign(-s7) + np.sign(-inv7) + np.sign(-cust7) + np.sign(-prod7)) / 4.0
    return x


def sample_weight_refined(y, future_ratio, profile):
    y = np.asarray(y, int)
    r = np.asarray(future_ratio, float)
    pos = max(int((y == 1).sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    w = np.where(y == 1, neg / pos, 1.0).astype(float)
    # Keep original 15% target. Only confidence of training examples changes.
    if profile == "baseline":
        margin = np.abs(r - .85)
        return w * np.clip(.55 + margin / .12, .55, 1.65)
    if profile == "soft_band":
        confidence = np.ones(len(r))
        confidence[(r > .82) & (r < .90)] = .35
        confidence[r <= .78] *= 1.25
        confidence[r >= .94] *= 1.10
        return w * confidence
    if profile == "strong_band":
        confidence = np.ones(len(r))
        confidence[(r > .80) & (r < .90)] = .20
        confidence[r <= .78] *= 1.40
        confidence[r >= .95] *= 1.15
        return w * confidence
    if profile == "asymmetric":
        confidence = np.ones(len(r))
        # Near-threshold negatives are the main false-positive source; downweight their label confidence.
        confidence[(y == 0) & (r >= .85) & (r < .92)] = .28
        confidence[(y == 1) & (r > .80)] = .45
        confidence[r <= .78] *= 1.35
        confidence[r >= .95] *= 1.10
        return w * confidence
    raise KeyError(profile)


def make_model(kind):
    if kind == "cb3":
        return CatBoostClassifier(iterations=650, depth=3, learning_rate=.020, l2_leaf_reg=16.0, random_seed=SEED, verbose=False, allow_writing_files=False, loss_function="Logloss", random_strength=.8)
    if kind == "cb4":
        return CatBoostClassifier(iterations=650, depth=4, learning_rate=.018, l2_leaf_reg=18.0, random_seed=SEED, verbose=False, allow_writing_files=False, loss_function="Logloss", random_strength=1.0)
    if kind == "cb5":
        return CatBoostClassifier(iterations=520, depth=5, learning_rate=.018, l2_leaf_reg=22.0, random_seed=SEED, verbose=False, allow_writing_files=False, loss_function="Logloss", random_strength=1.2)
    if kind == "xgb2":
        return XGBClassifier(n_estimators=620, max_depth=2, learning_rate=.018, min_child_weight=9, subsample=.86, colsample_bytree=.70, reg_alpha=2.0, reg_lambda=16.0, gamma=.20, objective="binary:logistic", eval_metric="logloss", random_state=SEED, n_jobs=2)
    raise KeyError(kind)


def candidate_oof(d, config):
    all_features = [c for c in d.columns if c not in META]
    parts = []
    fold_stats = []
    for fid, tr, va in v75.windows(d):
        Xtr0, Xva0 = v75.prepare(d.loc[tr, all_features], d.loc[va, all_features])
        ytr = d.loc[tr, "target"].astype(int)
        cols = v75.stable_top(Xtr0, ytr, config["topk"])
        Xtr = Xtr0[cols]
        Xva = Xva0[cols]
        model = make_model(config["model"])
        sw = sample_weight_refined(ytr, d.loc[tr, "future_ratio"], config["weight_profile"])
        model.fit(Xtr, ytr, sample_weight=sw)
        train_score = model.predict_proba(Xtr)[:, 1]
        val_score = model.predict_proba(Xva)[:, 1]
        rank = v75.percentile(val_score, train_score)
        yy = d.loc[va, "target"].astype(int).to_numpy()
        fold_stats.append({"fold_id": fid, "rows": int(va.sum()), "positives": int(yy.sum()), "roc_auc": float(roc_auc_score(yy, rank)), "pr_auc": float(average_precision_score(yy, rank)), "feature_count": len(cols)})
        parts.append(pd.DataFrame({"date": d.loc[va, "date"].to_numpy(), "target": yy, "fold_id": fid, "rank_score": rank, "raw_score": val_score}))
    o = pd.concat(parts, ignore_index=True).sort_values(["fold_id", "date"]).reset_index(drop=True)
    y = o.target.to_numpy(int)
    s = o.rank_score.to_numpy(float)
    return o, {"roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)), "min_fold_auc": float(min(x["roc_auc"] for x in fold_stats)), "mean_fold_pr": float(np.mean([x["pr_auc"] for x in fold_stats])), "folds": fold_stats}


def error_audit(d, diag, fused):
    # Evaluate V7.5 errors by future-ratio distance and selected behavior groups.
    e = fused.copy()
    panel_oof = []
    for fid, _, va in v75.windows(d):
        q = d.loc[va, ["date", "future_ratio"]].copy()
        q["fold_id"] = fid
        panel_oof.append(q)
    p = pd.concat(panel_oof, ignore_index=True).sort_values(["fold_id", "date"]).reset_index(drop=True)
    e = e.sort_values(["fold_id", "date"]).reset_index(drop=True)
    e["future_ratio"] = p.future_ratio.to_numpy(float)
    e["kind"] = np.select([(e.y == 1) & (e.v7_5_pred == 1), (e.y == 0) & (e.v7_5_pred == 1), (e.y == 1) & (e.v7_5_pred == 0)], ["TP", "FP", "FN"], default="TN")
    out = {"counts": e.kind.value_counts().to_dict(), "future_ratio_by_kind": {}}
    for kind in ["TP", "FP", "FN", "TN"]:
        z = e.loc[e.kind == kind, "future_ratio"]
        out["future_ratio_by_kind"][kind] = {"n": int(len(z)), "mean": float(z.mean()) if len(z) else None, "median": float(z.median()) if len(z) else None, "p10": float(z.quantile(.1)) if len(z) else None, "p90": float(z.quantile(.9)) if len(z) else None, "near_15pct_band_82_90_share": float(((z > .82) & (z < .90)).mean()) if len(z) else None}
    return out, e


def main():
    d = pd.read_csv(PANEL, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    d = add_hard_negative_features(d)
    diag = pd.read_csv(V61_DIAG)
    old = pd.read_csv(V75_FUSED)
    audit, audit_rows = error_audit(d, diag, old)
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    strong, quiet, market, base = v75.base_components(diag)
    y = diag.y.to_numpy(int)
    folds = diag.fold_id.to_numpy(int)
    mm = diag.merchant_mean.to_numpy(float)
    mp = diag.market_v3__risk_p90.to_numpy(float)
    base_metrics = v75.metrics(y, base, folds)
    if not np.array_equal(old.y.to_numpy(int), y):
        raise RuntimeError("V7.5 alignment mismatch")
    v75_metrics = v75.metrics(y, old.v7_5_pred.to_numpy(bool), folds)

    configs = []
    for model in ["cb3", "cb4", "cb5", "xgb2"]:
        for topk in [64, 96, 128]:
            for wp in ["baseline", "soft_band", "strong_band", "asymmetric"]:
                configs.append({"model": model, "topk": topk, "weight_profile": wp})

    candidates = []
    all_oof = []
    for i, cfg in enumerate(configs):
        o, rankm = candidate_oof(d, cfg)
        if len(o) != len(diag) or not np.array_equal(o.target.to_numpy(int), y) or not np.array_equal(o.fold_id.to_numpy(int), folds):
            raise RuntimeError(f"OOF alignment mismatch {cfg}")
        score = o.rank_score.to_numpy(float)
        pred, details = v75.prequential_fusion(y, folds, base, strong, quiet, market, score, mm, mp)
        fm = v75.metrics(y, pred, folds)
        strict = bool(fm["recall"] >= .80 and fm["precision"] > v75_metrics["precision"] and fm["f1"] > v75_metrics["f1"] and fm["green_npv"] >= .95 and fm["alert_rate"] <= v75_metrics["alert_rate"] and fm["worst_fold_recall"] >= .60 and fm["fp"] < v75_metrics["fp"])
        # Also measure an all-development ceiling for diagnosis only, never primary evidence.
        oracle = v75.oracle_ceiling(y, folds, base, strong, quiet, market, score, mm, mp, base_metrics)
        candidates.append({"config_id": i, "config": cfg, "ranking": rankm, "prequential_fusion": fm, "prequential_details": details, "strict_beats_v7_5": strict, "oracle_development_ceiling": oracle})
        z = o.copy(); z["config_id"] = i; all_oof.append(z)

    def key(c):
        m = c["prequential_fusion"]; r = c["ranking"]
        return (int(c["strict_beats_v7_5"]), m["f1"], m["precision"], -m["fp"], m["recall"], r["roc_auc"], r["pr_auc"])
    selected = max(candidates, key=key)
    sid = selected["config_id"]
    selected_oof = all_oof[sid].sort_values(["fold_id", "date"]).reset_index(drop=True)
    score = selected_oof.rank_score.to_numpy(float)
    final_pred, _ = v75.prequential_fusion(y, folds, base, strong, quiet, market, score, mm, mp)
    pd.concat(all_oof, ignore_index=True).to_csv(OOF, index=False)
    pd.DataFrame({"date": selected_oof.date, "y": y, "fold_id": folds, "v7_5_pred": old.v7_5_pred.to_numpy(int), "v8_rank": score, "v8_pred": final_pred.astype(int)}).to_csv(FUSED, index=False)

    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_BEST" if selected["strict_beats_v7_5"] else "EXPERIMENTAL_V7_5_REMAINS_BEST",
        "scientific_boundary": "V8 keeps the original 15% decline target for all outer evaluation. Ambiguity-zone handling affects training weights only; no ambiguous evaluation rows are removed. Hard-negative engineered features use current/past features only. Results remain development OOF evidence and require fresh Saudi merchant validation.",
        "rows": int(len(d)), "oof_rows": int(len(y)), "candidate_count": len(candidates), "added_hard_negative_features": 18,
        "v7_5_reference": v75_metrics, "v6_1_reference": base_metrics, "error_audit": audit,
        "selected": selected,
        "top_candidates": sorted(candidates, key=key, reverse=True)[:12],
        "target_definition": {"evaluation_decline": "future_ratio < 0.85", "training_ambiguity_only": "confidence downweighted around 0.80-0.90 depending on profile", "evaluation_rows_removed": 0},
        "red_supported": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    m = selected["prequential_fusion"]; r = selected["ranking"]
    lines = [
        "# Sales Sentinel V8 — Target Refinement + Hard-Negative Features", "",
        f"- Status: **{report['status']}**", f"- Candidates tested: **{len(candidates)}**",
        f"- Selected: **{selected['config']}**", "",
        "## Same-target prequential evidence",
        f"- ROC-AUC / PR-AUC: **{r['roc_auc']:.2%} / {r['pr_auc']:.2%}**",
        f"- Precision: V7.5 **{v75_metrics['precision']:.2%}** -> V8 **{m['precision']:.2%}**",
        f"- Recall: V7.5 **{v75_metrics['recall']:.2%}** -> V8 **{m['recall']:.2%}**",
        f"- F1: V7.5 **{v75_metrics['f1']:.2%}** -> V8 **{m['f1']:.2%}**",
        f"- NPV: V7.5 **{v75_metrics['green_npv']:.2%}** -> V8 **{m['green_npv']:.2%}**",
        f"- Alert rate: V7.5 **{v75_metrics['alert_rate']:.2%}** -> V8 **{m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Strictly beats V7.5: **{selected['strict_beats_v7_5']}**", "",
        "## Error audit",
        f"- V7.5 FP/FN: **{audit['counts'].get('FP',0)}/{audit['counts'].get('FN',0)}**",
        f"- FP share near 15% ambiguity band (future_ratio 0.82-0.90): **{audit['future_ratio_by_kind']['FP']['near_15pct_band_82_90_share']:.2%}**",
        "",
        "Scientific note: V8 did not remove ambiguous rows from evaluation; any gain is directly comparable with V7.5 on the original 15% target. External Saudi merchant validation is still pending."
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
