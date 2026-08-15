from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from xgboost import XGBRegressor

import train_merchant_error_corrector_v7_5 as v75
import train_merchant_target_refinement_v8 as v8

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V9-CONTINUOUS-FUTURE-RATIO-VERIFIER"
SEED = 42
PANEL = ROOT / "data" / "merchant_v7_1" / "merchant_feature_panel_v7_1.csv"
V61_DIAG = ROOT / "reports" / "merchant_market_fusion_v6_1" / "oof_policy_diagnostics.csv"
V82_OOF = ROOT / "reports" / "merchant_alert_verifier_v8_2" / "oof_predictions.csv"
OUT = ROOT / "reports" / "merchant_continuous_ratio_v9"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_candidate_predictions.csv"
SELECTED = OUT / "oof_selected_predictions.csv"
META = {"date", "future_ratio", "future7_sales", "baseline28_daily", "target"}


def stable_top_regression(X: pd.DataFrame, y: np.ndarray, n: int) -> list[str]:
    yy = np.asarray(y, float)
    cut = max(int(len(yy) * .55), 40)
    scores = []
    for c in X.columns:
        x = np.asarray(X[c], float)
        s1 = abs(np.corrcoef(x, yy)[0, 1]) if np.std(x) > 1e-12 and np.std(yy) > 1e-12 else 0.0
        xr, yr = x[-cut:], yy[-cut:]
        s2 = abs(np.corrcoef(xr, yr)[0, 1]) if np.std(xr) > 1e-12 and np.std(yr) > 1e-12 else 0.0
        if not np.isfinite(s1): s1 = 0.0
        if not np.isfinite(s2): s2 = 0.0
        scores.append((.65 * s1 + .35 * s2, c))
    scores.sort(reverse=True)
    return [c for _, c in scores[:n]]


def make_regressor(kind: str):
    if kind == "xgb_rmse":
        return XGBRegressor(
            n_estimators=650, max_depth=2, learning_rate=.018, min_child_weight=9,
            subsample=.86, colsample_bytree=.72, reg_alpha=2.0, reg_lambda=18.0,
            gamma=.15, objective="reg:squarederror", eval_metric="rmse",
            random_state=SEED, n_jobs=2,
        )
    if kind == "xgb_huber":
        return XGBRegressor(
            n_estimators=650, max_depth=2, learning_rate=.018, min_child_weight=9,
            subsample=.86, colsample_bytree=.72, reg_alpha=2.0, reg_lambda=18.0,
            gamma=.15, objective="reg:pseudohubererror", eval_metric="rmse",
            random_state=SEED, n_jobs=2,
        )
    if kind == "cb_rmse":
        return CatBoostRegressor(
            iterations=650, depth=4, learning_rate=.018, l2_leaf_reg=18.0,
            random_seed=SEED, verbose=False, allow_writing_files=False,
            loss_function="RMSE", random_strength=1.0,
        )
    if kind == "cb_q35":
        return CatBoostRegressor(
            iterations=650, depth=4, learning_rate=.018, l2_leaf_reg=18.0,
            random_seed=SEED, verbose=False, allow_writing_files=False,
            loss_function="Quantile:alpha=0.35", random_strength=1.0,
        )
    if kind == "cb_q50":
        return CatBoostRegressor(
            iterations=650, depth=4, learning_rate=.018, l2_leaf_reg=18.0,
            random_seed=SEED, verbose=False, allow_writing_files=False,
            loss_function="Quantile:alpha=0.50", random_strength=1.0,
        )
    raise KeyError(kind)


def transform_target(r: np.ndarray, mode: str) -> np.ndarray:
    r = np.clip(np.asarray(r, float), .35, 1.80)
    if mode == "ratio": return r
    if mode == "log_ratio": return np.log(r)
    raise KeyError(mode)


def inverse_target(z: np.ndarray, mode: str) -> np.ndarray:
    z = np.asarray(z, float)
    if mode == "ratio": return z
    if mode == "log_ratio": return np.exp(z)
    raise KeyError(mode)


def risk_percentile(pred_ratio: np.ndarray, train_pred_ratio: np.ndarray) -> np.ndarray:
    # Lower future ratio = higher decline risk.
    ref = np.sort(-np.asarray(train_pred_ratio, float))
    vals = -np.asarray(pred_ratio, float)
    return np.searchsorted(ref, vals, side="right") / max(len(ref), 1)


def candidate_oof(d: pd.DataFrame, cfg: dict):
    features = [c for c in d.columns if c not in META]
    parts, fold_stats = [], []
    for fid, tr, va in v75.windows(d):
        Xtr0, Xva0 = v75.prepare(d.loc[tr, features], d.loc[va, features])
        ytr_ratio = d.loc[tr, "future_ratio"].to_numpy(float)
        cols = stable_top_regression(Xtr0, transform_target(ytr_ratio, cfg["target_mode"]), cfg["topk"])
        Xtr, Xva = Xtr0[cols], Xva0[cols]
        model = make_regressor(cfg["model"])
        yfit = transform_target(ytr_ratio, cfg["target_mode"])
        # Stronger weight around genuine future declines without discarding ambiguous cases.
        sw = 1.0 + 1.4 * (ytr_ratio < .85).astype(float) + .35 * np.clip(np.abs(ytr_ratio - .85) / .20, 0, 2)
        model.fit(Xtr, yfit, sample_weight=sw)
        tr_pred = inverse_target(model.predict(Xtr), cfg["target_mode"])
        va_pred = inverse_target(model.predict(Xva), cfg["target_mode"])
        risk = risk_percentile(va_pred, tr_pred)
        yy = d.loc[va, "target"].astype(int).to_numpy()
        actual = d.loc[va, "future_ratio"].to_numpy(float)
        fold_stats.append({
            "fold_id": int(fid), "rows": int(va.sum()), "positives": int(yy.sum()),
            "mae_ratio": float(mean_absolute_error(actual, va_pred)),
            "rmse_ratio": float(mean_squared_error(actual, va_pred) ** .5),
            "roc_auc": float(roc_auc_score(yy, risk)),
            "pr_auc": float(average_precision_score(yy, risk)),
            "feature_count": int(len(cols)),
        })
        parts.append(pd.DataFrame({
            "date": d.loc[va, "date"].to_numpy(), "target": yy, "fold_id": fid,
            "actual_future_ratio": actual, "pred_future_ratio": va_pred, "risk_score": risk,
        }))
    o = pd.concat(parts, ignore_index=True).sort_values(["fold_id", "date"]).reset_index(drop=True)
    y = o.target.to_numpy(int); s = o.risk_score.to_numpy(float)
    return o, {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "mae_ratio": float(mean_absolute_error(o.actual_future_ratio, o.pred_future_ratio)),
        "rmse_ratio": float(mean_squared_error(o.actual_future_ratio, o.pred_future_ratio) ** .5),
        "min_fold_auc": float(min(x["roc_auc"] for x in fold_stats)),
        "folds": fold_stats,
    }


def apply_rule(base: np.ndarray, strong: np.ndarray, risk: np.ndarray, pred_ratio: np.ndarray, rule: tuple):
    pred = np.asarray(base, bool).copy()
    scope, risk_veto, ratio_veto, rescue = rule
    if scope != "none":
        if scope == "nonstrong": mask = ~strong
        else: mask = np.ones(len(pred), bool)
        # Veto only when both continuous views say risk is weak / rebound plausible.
        veto = pred & mask & (risk < risk_veto) & (pred_ratio > ratio_veto)
        pred[veto] = False
    if rescue <= 1.0:
        # Conservative rescue only at very high regression-derived risk.
        pred = pred | ((~base) & (risk >= rescue) & (pred_ratio < .82))
    return pred


def rule_grid():
    out = []
    for scope, rv, ratio, rescue in product(
        ["none", "nonstrong", "any"],
        [0.0, .10, .20, .30, .40, .50, .60],
        [.90, .95, 1.00, 1.05],
        [1.01, .90, .95],
    ):
        if scope == "none" and rv != 0.0: continue
        if scope != "none" and rv == 0.0: continue
        out.append((scope, rv, ratio, rescue))
    return out


def prequential(y, folds, base, strong, risk, pred_ratio):
    final = np.asarray(base, bool).copy(); details = []
    for f in sorted(np.unique(folds)):
        cur = folds == f
        if f == 0:
            details.append({"fold_id": int(f), "mode": "v8_2_bootstrap"})
            continue
        hist = folds < f
        bm = v75.metrics(y[hist], base[hist], folds[hist])
        best = None
        for rule in rule_grid():
            p = apply_rule(base[hist], strong[hist], risk[hist], pred_ratio[hist], rule)
            m = v75.metrics(y[hist], p, folds[hist])
            feasible = (
                m["recall"] >= bm["recall"] and
                m["green_npv"] >= bm["green_npv"] - .002 and
                m["fp"] <= bm["fp"] and
                m["f1"] >= bm["f1"] and
                m["alert_rate"] <= bm["alert_rate"]
            )
            key = (int(feasible), m["f1"], m["precision"], -m["fp"], m["recall"], -m["alert_rate"])
            if best is None or key > best[0]: best = (key, rule, m, feasible)
        _, rule, hm, feasible = best
        if not feasible: rule = ("none", 0.0, .95, 1.01)
        final[cur] = apply_rule(base[cur], strong[cur], risk[cur], pred_ratio[cur], rule)[cur]
        details.append({"fold_id": int(f), "history_rule": list(rule), "history_feasible": bool(feasible), "history_metrics": hm})
    return final, details


def main():
    d = pd.read_csv(PANEL, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    d = v8.add_hard_negative_features(d)
    diag = pd.read_csv(V61_DIAG)
    base_oof = pd.read_csv(V82_OOF).sort_values(["fold_id", "date"]).reset_index(drop=True)
    y = diag.y.to_numpy(int); folds = diag.fold_id.to_numpy(int)
    if len(base_oof) != len(y) or not np.array_equal(base_oof.y.to_numpy(int), y):
        raise RuntimeError("V8.2 OOF alignment mismatch")
    base = base_oof.v8_2_pred.to_numpy(bool)
    strong, _, _, _ = v75.base_components(diag)
    base_metrics = v75.metrics(y, base, folds)

    configs = []
    for model in ["xgb_rmse", "xgb_huber", "cb_rmse", "cb_q35", "cb_q50"]:
        for topk in [64, 96, 128]:
            for target_mode in ["ratio", "log_ratio"]:
                configs.append({"model": model, "topk": topk, "target_mode": target_mode})

    candidates, all_parts = [], []
    for cid, cfg in enumerate(configs):
        o, rankm = candidate_oof(d, cfg)
        if len(o) != len(y) or not np.array_equal(o.target.to_numpy(int), y) or not np.array_equal(o.fold_id.to_numpy(int), folds):
            raise RuntimeError(f"OOF alignment mismatch: {cfg}")
        pred, details = prequential(y, folds, base, strong, o.risk_score.to_numpy(float), o.pred_future_ratio.to_numpy(float))
        m = v75.metrics(y, pred, folds)
        strict = bool(
            m["recall"] >= base_metrics["recall"] and
            m["precision"] > base_metrics["precision"] and
            m["f1"] > base_metrics["f1"] and
            m["green_npv"] >= base_metrics["green_npv"] and
            m["alert_rate"] <= base_metrics["alert_rate"] and
            m["fp"] < base_metrics["fp"] and
            m["worst_fold_recall"] >= base_metrics["worst_fold_recall"]
        )
        candidates.append({
            "config_id": cid, "config": cfg, "regression_ranking": rankm,
            "prequential_metrics": m, "prequential_details": details,
            "strictly_dominates_v8_2": strict,
        })
        z = o.copy(); z["config_id"] = cid; all_parts.append(z)

    def key(c):
        m = c["prequential_metrics"]; r = c["regression_ranking"]
        return (int(c["strictly_dominates_v8_2"]), m["f1"], m["precision"], -m["fp"], m["recall"], r["roc_auc"], -r["mae_ratio"])

    selected = max(candidates, key=key)
    sid = selected["config_id"]
    sel = all_parts[sid].sort_values(["fold_id", "date"]).reset_index(drop=True)
    final_pred, _ = prequential(y, folds, base, strong, sel.risk_score.to_numpy(float), sel.pred_future_ratio.to_numpy(float))
    pd.concat(all_parts, ignore_index=True).to_csv(OOF, index=False)
    pd.DataFrame({
        "date": sel.date, "y": y, "fold_id": folds, "v8_2_pred": base.astype(int),
        "actual_future_ratio": sel.actual_future_ratio, "pred_future_ratio": sel.pred_future_ratio,
        "v9_risk": sel.risk_score, "v9_pred": final_pred.astype(int),
    }).to_csv(SELECTED, index=False)

    report = {
        "version": VERSION,
        "status": "DEVELOPMENT_BEST" if selected["strictly_dominates_v8_2"] else "EXPERIMENTAL_V8_2_REMAINS_BEST",
        "scientific_boundary": "V9 uses continuous future-ratio regression as a causal prequential verifier over frozen V8.2 OOF decisions. Model/config selection still occurs on previously used development folds; external Saudi merchant validation remains required.",
        "candidate_count": len(candidates),
        "v8_2": base_metrics,
        "selected": selected,
        "top_candidates": sorted(candidates, key=key, reverse=True)[:8],
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    m = selected["prequential_metrics"]; r = selected["regression_ranking"]
    lines = [
        "# Sales Sentinel V9 — Continuous Future-Ratio Verifier", "",
        f"- Status: **{report['status']}**",
        f"- Candidates: **{len(candidates)}**",
        f"- Selected: **{selected['config']}**", "",
        f"- Ratio MAE / RMSE: **{r['mae_ratio']:.4f} / {r['rmse_ratio']:.4f}**",
        f"- Regression-derived ROC-AUC / PR-AUC: **{r['roc_auc']:.2%} / {r['pr_auc']:.2%}**", "",
        f"- Precision: V8.2 **{base_metrics['precision']:.2%}** -> V9 **{m['precision']:.2%}**",
        f"- Recall: V8.2 **{base_metrics['recall']:.2%}** -> V9 **{m['recall']:.2%}**",
        f"- F1: V8.2 **{base_metrics['f1']:.2%}** -> V9 **{m['f1']:.2%}**",
        f"- NPV: V8.2 **{base_metrics['green_npv']:.2%}** -> V9 **{m['green_npv']:.2%}**",
        f"- Alert rate: V8.2 **{base_metrics['alert_rate']:.2%}** -> V9 **{m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",
        f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",
        f"- Strictly dominates V8.2: **{selected['strictly_dominates_v8_2']}**", "",
        "Scientific boundary: development evidence only; external real Saudi merchant validation is still pending.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
