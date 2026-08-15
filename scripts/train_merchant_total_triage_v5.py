from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_merchant_total_hybrid_v4_3 as v3

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-MERCHANT-TOTAL-TRIAGE-5.0"
SRC = ROOT / "data" / "merchant_v4_3" / "merchant_total_feature_panel_v4_3.csv"
BASELINE_REPORT = ROOT / "reports" / "merchant_total_triage_v4_5" / "development_report.json"
OUT = ROOT / "reports" / "merchant_total_triage_v5"
MOD = ROOT / "models" / "merchant_total_triage_v5"
OUT.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
MODEL = MOD / "merchant_total_triage_v5.joblib"
SEED = 42
EARLY_RATIO = 0.85
SEVERE_RATIO = 0.80
MAX_FEATURE_BUDGET = 72


def model_factories():
    # Selection remains inside each Pipeline, so every OOF fold selects features
    # using its historical training side only.
    return {
        "logreg_k36": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=36)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.08, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=SEED)),
        ]),
        "logreg_k60": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=60)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=SEED)),
        ]),
        "extra_k48": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=48)),
            ("model", ExtraTreesClassifier(n_estimators=1200, max_depth=5, min_samples_leaf=7, max_features=0.65, class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ]),
        "extra_k72": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif, k=72)),
            ("model", ExtraTreesClassifier(n_estimators=1400, max_depth=6, min_samples_leaf=8, max_features=0.55, class_weight="balanced", random_state=SEED + 1, n_jobs=-1)),
        ]),
    }


def metrics(y, score, threshold):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score >= float(threshold)
    tn = int(((y == 0) & (~pred)).sum())
    fn = int(((y == 1) & (~pred)).sum())
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "alert_rate": float(pred.mean()),
        "green_npv": float(tn / max(tn + fn, 1)),
        "tp": int(((y == 1) & pred).sum()),
        "fp": int(((y == 0) & pred).sum()),
        "fn": fn,
        "tn": tn,
    }


def fold_metrics(y, score, threshold, fold_ids):
    out = []
    for fid in sorted(np.unique(fold_ids)):
        mask = fold_ids == fid
        m = metrics(y[mask], score[mask], threshold)
        m["fold_id"] = int(fid)
        m["rows"] = int(mask.sum())
        m["positives"] = int(y[mask].sum())
        out.append(m)
    return out


def choose_early_threshold(y, score, fold_ids):
    cand = np.unique(np.r_[np.linspace(0.05, 0.95, 181), np.quantile(score, np.linspace(0.02, 0.98, 121))])
    rows = []
    for t in cand:
        m = metrics(y, score, t)
        per = fold_metrics(y, score, t, fold_ids)
        worst_recall = min((z["recall"] for z in per if z["positives"] >= 5), default=m["recall"])
        max_alert = max((z["alert_rate"] for z in per), default=m["alert_rate"])
        supported = (
            m["recall"] >= 0.78
            and m["precision"] >= 0.30
            and m["green_npv"] >= 0.94
            and m["alert_rate"] <= 0.45
            and worst_recall >= 0.40
            and max_alert <= 0.70
        )
        objective = 1.20 * m["f1"] + 0.55 * m["balanced_accuracy"] + 0.35 * worst_recall + 0.20 * m["green_npv"] - 0.25 * m["alert_rate"]
        rows.append((supported, objective, float(t), m, per, worst_recall, max_alert))
    pool = [r for r in rows if r[0]] or rows
    pool.sort(key=lambda r: (r[0], r[1], r[3]["recall"], r[3]["precision"]), reverse=True)
    best = pool[0]
    return {
        "supported": bool(best[0]),
        "threshold": best[2],
        "metrics": best[3],
        "per_fold": best[4],
        "worst_fold_recall": float(best[5]),
        "max_fold_alert_rate": float(best[6]),
        "feasible_thresholds": int(sum(1 for r in rows if r[0])),
        "objective": float(best[1]),
    }


def choose_red_threshold(y, score, fold_ids):
    cand = np.unique(np.r_[np.linspace(0.15, 0.98, 167), np.quantile(score, np.linspace(0.40, 0.995, 120))])
    rows = []
    for t in cand:
        m = metrics(y, score, t)
        per = fold_metrics(y, score, t, fold_ids)
        alerts = m["tp"] + m["fp"]
        supported = alerts >= 5 and m["precision"] >= 0.60 and m["recall"] >= 0.20
        objective = 1.10 * m["f1"] + 0.70 * m["precision"] + 0.40 * m["recall"] - 0.15 * m["alert_rate"]
        rows.append((supported, objective, float(t), m, per))
    feasible = [r for r in rows if r[0]]
    if not feasible:
        return {
            "supported": False,
            "threshold": None,
            "metrics": metrics(y, score, float("inf")),
            "per_fold": fold_metrics(y, score, float("inf"), fold_ids),
            "feasible_thresholds": 0,
        }
    feasible.sort(key=lambda r: (r[1], r[3]["precision"], r[3]["recall"]), reverse=True)
    best = feasible[0]
    return {
        "supported": True,
        "threshold": best[2],
        "metrics": best[3],
        "per_fold": best[4],
        "feasible_thresholds": len(feasible),
    }


def oof_for_target(X, y, folds):
    result = {}
    for name, factory in model_factories().items():
        score = np.full(len(X), np.nan, dtype=float)
        fold_id = np.full(len(X), -1, dtype=int)
        fold_meta = []
        for fid, (st, en, tr, va) in enumerate(folds):
            ytr = y[tr.to_numpy()]
            if len(np.unique(ytr)) < 2:
                raise RuntimeError(f"{name}: fold {fid} training data has one class")
            model = clone(factory)
            model.fit(X.loc[tr], ytr)
            idx = np.where(va.to_numpy())[0]
            score[idx] = model.predict_proba(X.loc[va])[:, 1]
            fold_id[idx] = fid
            fold_meta.append({
                "fold_id": fid,
                "start": str(st.date()),
                "end": str(en.date()),
                "train_rows": int(tr.sum()),
                "validation_rows": int(va.sum()),
                "train_positives": int(ytr.sum()),
                "validation_positives": int(y[va.to_numpy()].sum()),
            })
        mask = fold_id >= 0
        result[name] = {"score": score[mask], "y": y[mask], "fold_id": fold_id[mask], "mask": mask, "folds": fold_meta}
    return result


def selected_feature_names(pipe, feature_names):
    names = np.asarray(feature_names, dtype=object)
    names = names[pipe.named_steps["variance"].get_support()]
    return [str(x) for x in names[pipe.named_steps["select"].get_support()]]


def load_baseline():
    if not BASELINE_REPORT.exists():
        return None
    try:
        return json.loads(BASELINE_REPORT.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    d = pd.read_csv(SRC, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if not {"date", "future_ratio", "target"}.issubset(d.columns):
        raise RuntimeError("merchant_total_feature_panel_v4_3.csv is missing required columns")
    meta = d[["date", "future_ratio"]].copy()
    X = d.drop(columns=["date", "future_ratio", "target"]).replace([np.inf, -np.inf], np.nan)
    y15_all = (meta.future_ratio.to_numpy(float) < EARLY_RATIO).astype(int)
    y20_all = (meta.future_ratio.to_numpy(float) < SEVERE_RATIO).astype(int)

    # v3.folds uses expanding rolling origin and a seven-day purge before validation.
    folds = v3.folds(meta.assign(target=y20_all))
    if len(folds) != 5:
        raise RuntimeError(f"Expected five purged rolling folds, got {len(folds)}")

    early_oof = oof_for_target(X, y15_all, folds)
    early_candidates = {}
    for name, z in early_oof.items():
        sel = choose_early_threshold(z["y"], z["score"], z["fold_id"])
        sel["roc_auc"] = float(roc_auc_score(z["y"], z["score"]))
        early_candidates[name] = sel
    early_name = max(early_candidates, key=lambda n: (early_candidates[n]["supported"], early_candidates[n]["objective"], early_candidates[n]["metrics"]["f1"], early_candidates[n]["roc_auc"]))
    early_sel = early_candidates[early_name]

    severe_oof = oof_for_target(X, y20_all, folds)
    severe_candidates = {}
    for name, z in severe_oof.items():
        sel = choose_red_threshold(z["y"], z["score"], z["fold_id"])
        sel["roc_auc"] = float(roc_auc_score(z["y"], z["score"]))
        severe_candidates[name] = sel
    red_name = max(severe_candidates, key=lambda n: (severe_candidates[n]["supported"], severe_candidates[n]["metrics"]["f1"], severe_candidates[n]["metrics"]["precision"], severe_candidates[n]["roc_auc"]))
    red_sel = severe_candidates[red_name]

    early_model = clone(model_factories()[early_name]).fit(X, y15_all)
    red_model = clone(model_factories()[red_name]).fit(X, y20_all)
    early_features = selected_feature_names(early_model, X.columns)
    red_features = selected_feature_names(red_model, X.columns)

    artifact = {
        "version": VERSION,
        "status": "DEVELOPMENT_FROZEN_PENDING_EXTERNAL_VALIDATION",
        "target_definition": "next 7-day sales total / (7 * trailing 28-day daily mean)",
        "early_decline_ratio": EARLY_RATIO,
        "severe_decline_ratio": SEVERE_RATIO,
        "feature_selection_policy": "train-fold-only SelectKBest(f_classif) inside sklearn Pipeline",
        "threshold_policy": "single threshold selected from purged rolling OOF only",
        "early_model_name": early_name,
        "red_model_name": red_name,
        "early_model": early_model,
        "red_model": red_model,
        "early_threshold": float(early_sel["threshold"]),
        "red_threshold": float(red_sel["threshold"]) if red_sel["supported"] else None,
        "early_feature_columns": early_features,
        "red_feature_columns": red_features,
        "input_feature_columns": list(X.columns),
        "states": {
            "GREEN": "early probability below frozen OOF threshold",
            "AMBER": ">=15% decline warning",
            "RED": ">=20% severe decline warning only when high-precision OOF support exists",
        },
    }
    joblib.dump(artifact, MODEL)

    baseline = load_baseline()
    base_m = (baseline or {}).get("combined15", {})
    m5 = early_sel["metrics"]
    comparison = {
        "baseline_version": (baseline or {}).get("version"),
        "same_panel_rows": int(len(d)),
        "v4_5": {k: base_m.get(k) for k in ["precision", "recall", "f1", "alert_rate", "green_npv"]},
        "v5": {k: m5.get(k) for k in ["precision", "recall", "f1", "alert_rate", "green_npv", "balanced_accuracy", "roc_auc"]},
        "delta": {k: (float(m5[k]) - float(base_m[k])) if base_m.get(k) is not None else None for k in ["precision", "recall", "f1", "alert_rate", "green_npv"]},
    }

    contract = {
        "early_recall_min": 0.78,
        "early_precision_min": 0.30,
        "early_green_npv_min": 0.94,
        "early_alert_rate_max": 0.45,
        "worst_fold_recall_min": 0.40,
        "feature_budget_max": MAX_FEATURE_BUDGET,
    }
    gates = {
        "rolling_origin_past_only": True,
        "target_purge_7days": True,
        "feature_selection_inside_fold": True,
        "threshold_selected_from_oof_only": True,
        "no_synthetic_oversampling": True,
        "feature_budget": len(early_features) <= MAX_FEATURE_BUDGET,
        "early_recall": m5["recall"] >= contract["early_recall_min"],
        "early_precision": m5["precision"] >= contract["early_precision_min"],
        "early_green_npv": m5["green_npv"] >= contract["early_green_npv_min"],
        "early_alert_rate": m5["alert_rate"] <= contract["early_alert_rate_max"],
        "worst_fold_recall": early_sel["worst_fold_recall"] >= contract["worst_fold_recall_min"],
        "early_threshold_supported": bool(early_sel["supported"]),
    }

    report = {
        "version": VERSION,
        "status": artifact["status"],
        "scientific_boundary": "V5 is rolling-origin development evidence on the same localized merchant panel. It fixes high-dimensional training and threshold leakage risks, but it is not an external Saudi merchant validation and must not be reported as independent real-world accuracy.",
        "rows": int(len(d)),
        "raw_feature_count": int(X.shape[1]),
        "selected_early_feature_count": len(early_features),
        "selected_red_feature_count": len(red_features),
        "oof_rows": int(len(early_oof[early_name]["y"])),
        "positive_rate_early15": float(y15_all.mean()),
        "positive_rate_severe20": float(y20_all.mean()),
        "early_model": early_name,
        "red_model": red_name,
        "early_threshold": early_sel,
        "red_threshold": red_sel,
        "early_candidates": early_candidates,
        "red_candidates": severe_candidates,
        "selected_early_features": early_features,
        "selected_red_features": red_features,
        "folds": early_oof[early_name]["folds"],
        "comparison_vs_v4_5": comparison,
        "contract": contract,
        "gates": gates,
        "all_development_gates_passed": bool(all(gates.values())),
        "next_required_evidence": "Freeze V5, run a new non-reused stress suite, then validate on longitudinal real Saudi merchant data.",
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    delta_f1 = comparison["delta"].get("f1")
    delta_recall = comparison["delta"].get("recall")
    summary_lines = [
        "# Sales Sentinel v5.0 — Leakage-safe feature-selection triage",
        "",
        f"- Raw features: **{X.shape[1]}**",
        f"- Selected early features: **{len(early_features)}**",
        f"- Early model: **{early_name}**",
        f"- Early ROC-AUC: **{early_sel['roc_auc']:.2%}**",
        f"- Early precision: **{m5['precision']:.2%}**",
        f"- Early recall: **{m5['recall']:.2%}**",
        f"- Early F1: **{m5['f1']:.2%}**",
        f"- GREEN NPV: **{m5['green_npv']:.2%}**",
        f"- Alert rate: **{m5['alert_rate']:.2%}**",
        f"- Worst-fold recall: **{early_sel['worst_fold_recall']:.2%}**",
        f"- RED supported: **{red_sel['supported']}**",
    ]
    if delta_f1 is not None:
        summary_lines.append(f"- F1 delta vs v4.5: **{delta_f1:+.2%}**")
    if delta_recall is not None:
        summary_lines.append(f"- Recall delta vs v4.5: **{delta_recall:+.2%}**")
    summary_lines.append(f"- Development gates: **{all(gates.values())}**")
    SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
