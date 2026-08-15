from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

import train_merchant_error_corrector_v7_5 as v75
import train_merchant_target_refinement_v8 as v8

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V10-SEVERITY-AWARE-DECLINE"
PANEL = ROOT / "data" / "merchant_v7_1" / "merchant_feature_panel_v7_1.csv"
BASE_OOF = ROOT / "reports" / "merchant_meta_verifier_v9_2" / "oof_predictions.csv"
DIAG = ROOT / "reports" / "merchant_market_fusion_v6_1" / "oof_policy_diagnostics.csv"
OUT = ROOT / "reports" / "merchant_severity_aware_v10"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_candidate_scores.csv"
SELECTED = OUT / "oof_selected_predictions.csv"
META = {"date", "future_ratio", "future7_sales", "baseline28_daily", "target"}
SEED = 42


def make_classes(r, boundary, growth_split):
    r = np.asarray(r, float)
    if growth_split is None:
        # 0 severe, 1 moderate/borderline, 2 normal/rebound
        return np.select([r < boundary, r < .85], [0, 1], default=2).astype(int)
    # 0 severe, 1 moderate, 2 normal, 3 strong rebound/growth
    return np.select([r < boundary, r < .85, r < growth_split], [0, 1, 2], default=3).astype(int)


def make_model(kind, nclass):
    if kind == "catboost":
        return CatBoostClassifier(
            iterations=650, depth=4, learning_rate=.018, l2_leaf_reg=18.0,
            random_seed=SEED, verbose=False, allow_writing_files=False,
            loss_function="MultiClass", auto_class_weights="Balanced", random_strength=1.0,
        )
    if kind == "xgb":
        return XGBClassifier(
            n_estimators=620, max_depth=2, learning_rate=.018, min_child_weight=8,
            subsample=.86, colsample_bytree=.72, reg_alpha=2.0, reg_lambda=16.0,
            gamma=.18, objective="multi:softprob", num_class=nclass,
            eval_metric="mlogloss", random_state=SEED, n_jobs=2,
        )
    raise KeyError(kind)


def class_weights(y):
    y = np.asarray(y, int)
    counts = np.bincount(y)
    n = len(y); k = len(counts)
    return np.array([n / max(k * counts[c], 1) for c in y], float)


def candidate_oof(d, cfg):
    features = [c for c in d.columns if c not in META]
    parts, folds_out = [], []
    for fid, tr, va in v75.windows(d):
        Xtr0, Xva0 = v75.prepare(d.loc[tr, features], d.loc[va, features])
        ybin = d.loc[tr, "target"].astype(int)
        cols = v75.stable_top(Xtr0, ybin, cfg["topk"])
        Xtr, Xva = Xtr0[cols], Xva0[cols]
        ytr = make_classes(d.loc[tr, "future_ratio"].to_numpy(float), cfg["boundary"], cfg["growth_split"])
        nclass = int(ytr.max()) + 1
        model = make_model(cfg["model"], nclass)
        if cfg["model"] == "xgb":
            model.fit(Xtr, ytr, sample_weight=class_weights(ytr))
        else:
            model.fit(Xtr, ytr)
        proba = model.predict_proba(Xva)
        # Decline risk = severe + moderate probability.
        risk = proba[:, 0] + proba[:, 1]
        pred_class = np.argmax(proba, axis=1)
        yy = d.loc[va, "target"].astype(int).to_numpy()
        folds_out.append({
            "fold_id": int(fid), "rows": int(va.sum()), "positives": int(yy.sum()),
            "roc_auc": float(roc_auc_score(yy, risk)),
            "pr_auc": float(average_precision_score(yy, risk)),
            "feature_count": len(cols), "nclass": nclass,
        })
        parts.append(pd.DataFrame({
            "date": d.loc[va, "date"].to_numpy(), "target": yy, "fold_id": fid,
            "severity_risk": risk, "pred_class": pred_class,
        }))
    o = pd.concat(parts, ignore_index=True).sort_values(["fold_id", "date"]).reset_index(drop=True)
    y = o.target.to_numpy(int); s = o.severity_risk.to_numpy(float)
    return o, {
        "roc_auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s)),
        "min_fold_auc": float(min(z["roc_auc"] for z in folds_out)), "folds": folds_out,
    }


def apply_rule(base, strong, risk, pred_class, rule):
    scope, risk_veto, class_veto = rule
    pred = np.asarray(base, bool).copy()
    if scope == "none": return pred
    mask = (~strong) if scope == "nonstrong" else np.ones(len(pred), bool)
    # Only veto when the multiclass model says normal/rebound and decline probability is low.
    veto = pred & mask & (risk < risk_veto) & (pred_class >= class_veto)
    pred[veto] = False
    return pred


def rule_grid():
    out = [("none", 0.0, 2)]
    for scope, rv, cv in product(["nonstrong", "any"], [.15, .20, .25, .30, .35, .40, .45, .50, .55], [2, 3]):
        out.append((scope, rv, cv))
    return out


def prequential(y, folds, base, strong, risk, pred_class):
    final = np.asarray(base, bool).copy(); details = []
    for f in sorted(np.unique(folds)):
        cur = folds == f
        if f == 0:
            details.append({"fold_id": int(f), "mode": "v9_2_bootstrap"}); continue
        hist = folds < f
        bm = v75.metrics(y[hist], base[hist], folds[hist])
        best = None
        for rule in rule_grid():
            p = apply_rule(base[hist], strong[hist], risk[hist], pred_class[hist], rule)
            m = v75.metrics(y[hist], p, folds[hist])
            feasible = (
                m["recall"] >= bm["recall"] and m["green_npv"] >= bm["green_npv"] - .001 and
                m["fp"] <= bm["fp"] and m["f1"] >= bm["f1"] and m["alert_rate"] <= bm["alert_rate"]
            )
            key = (int(feasible), m["f1"], m["precision"], -m["fp"], m["recall"])
            if best is None or key > best[0]: best = (key, rule, m, feasible)
        _, rule, hm, feasible = best
        if not feasible: rule = ("none", 0.0, 2)
        final[cur] = apply_rule(base[cur], strong[cur], risk[cur], pred_class[cur], rule)
        details.append({"fold_id": int(f), "history_rule": list(rule), "history_feasible": bool(feasible), "history_metrics": hm})
    return final, details


def main():
    d = pd.read_csv(PANEL, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    d = v8.add_hard_negative_features(d)
    base_oof = pd.read_csv(BASE_OOF).sort_values(["fold_id", "date"]).reset_index(drop=True)
    diag = pd.read_csv(DIAG)
    y = diag.y.to_numpy(int); folds = diag.fold_id.to_numpy(int)
    if len(base_oof) != len(y) or not np.array_equal(base_oof.y.to_numpy(int), y):
        raise RuntimeError("V9.2 alignment mismatch")
    base = base_oof.v9_2_pred.to_numpy(bool)
    strong, _, _, _ = v75.base_components(diag)
    bm = v75.metrics(y, base, folds)

    configs = []
    for model, topk, boundary, growth_split in product(
        ["catboost", "xgb"], [64, 96, 128], [.75, .78, .80], [None, 1.05]
    ):
        configs.append({"model": model, "topk": topk, "boundary": boundary, "growth_split": growth_split})

    rows, all_oof, preds = [], [], []
    for cid, cfg in enumerate(configs):
        o, rankm = candidate_oof(d, cfg)
        if len(o) != len(y) or not np.array_equal(o.target.to_numpy(int), y):
            raise RuntimeError(f"OOF mismatch {cfg}")
        p, details = prequential(y, folds, base, strong, o.severity_risk.to_numpy(float), o.pred_class.to_numpy(int))
        m = v75.metrics(y, p, folds)
        adopt = bool(
            m["recall"] >= bm["recall"] and m["green_npv"] >= bm["green_npv"] and
            m["precision"] > bm["precision"] and m["f1"] > bm["f1"] and
            m["fp"] < bm["fp"] and m["worst_fold_recall"] >= bm["worst_fold_recall"] and
            m["alert_rate"] <= bm["alert_rate"]
        )
        rows.append({"config_id": cid, "config": cfg, "ranking": rankm, "metrics": m, "strictly_dominates_v9_2": adopt, "details": details})
        z = o.copy(); z["config_id"] = cid; all_oof.append(z); preds.append(p)

    def key(r):
        m = r["metrics"]; q = r["ranking"]
        return (int(r["strictly_dominates_v9_2"]), m["f1"], m["precision"], -m["fp"], m["recall"], q["roc_auc"], q["pr_auc"])
    sel = max(rows, key=key); sid = sel["config_id"]; p = preds[sid]; m = sel["metrics"]
    so = all_oof[sid].sort_values(["fold_id", "date"]).reset_index(drop=True)
    pd.concat(all_oof, ignore_index=True).to_csv(OOF, index=False)
    pd.DataFrame({
        "date": so.date, "y": y, "fold_id": folds, "v9_2_pred": base.astype(int),
        "severity_risk": so.severity_risk, "pred_class": so.pred_class, "v10_pred": p.astype(int),
    }).to_csv(SELECTED, index=False)

    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_BEST" if sel["strictly_dominates_v9_2"] else "EXPERIMENTAL_V9_2_REMAINS_BEST",
        "scientific_boundary": "V10 severity-aware models are evaluated with rolling-origin OOF and causal earlier-fold rule selection. Configuration selection is still development evidence; external Saudi merchant validation remains required.",
        "candidate_count": len(rows), "v9_2": bm, "selected": sel,
        "top_candidates": sorted(rows, key=key, reverse=True)[:10], "red_supported": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    q = sel["ranking"]
    lines = [
        "# Sales Sentinel V10 — Severity-Aware Decline Model", "",
        f"- Status: **{report['status']}**", f"- Candidates: **{len(rows)}**",
        f"- Selected: **{sel['config']}**", "",
        f"- Severity ROC-AUC / PR-AUC: **{q['roc_auc']:.2%} / {q['pr_auc']:.2%}**",
        f"- Precision: V9.2 **{bm['precision']:.2%}** -> V10 **{m['precision']:.2%}**",
        f"- Recall: V9.2 **{bm['recall']:.2%}** -> V10 **{m['recall']:.2%}**",
        f"- F1: V9.2 **{bm['f1']:.2%}** -> V10 **{m['f1']:.2%}**",
        f"- NPV: V9.2 **{bm['green_npv']:.2%}** -> V10 **{m['green_npv']:.2%}**",
        f"- Alert rate: V9.2 **{bm['alert_rate']:.2%}** -> V10 **{m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Strictly dominates V9.2: **{sel['strictly_dominates_v9_2']}**", "",
        "Scientific boundary: development evidence only; external Saudi merchant validation remains pending.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
