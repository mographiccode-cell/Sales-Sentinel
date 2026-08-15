from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

import train_merchant_category_signals_v7_1 as v71
import evaluate_redsea_portable_v16 as v16

ROOT = Path(__file__).resolve().parents[1]
REDSEA_FILE = Path(os.environ.get("REDSEA_FILE", "/tmp/redsea_mendeley/RedSea_Data_Cleaned.xlsx"))
OUT = ROOT / "reports" / "redsea_portable_v16_1"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "diagnostic_report.json"
SUMMARY = OUT / "summary.md"
DEV_OOF = OUT / "development_oof.csv"
REDSEA_PRED = OUT / "redsea_predictions.csv"
DRIFT = OUT / "feature_drift.csv"

VERSION = "SALES-SENTINEL-V16.1-COMPARABILITY-ABLATION"
SEED = 20260816

# These exclusions are semantic comparability exclusions, not Redsea-label optimization:
# - new/returning customer status depends on the start of each observed history window;
#   Redsea begins only in July 2023, so first-seen status is not comparable with long source history.
# - year-cycle sin/cos is poorly identified by a four-month external window and is not needed for
#   daily/weekly Saudi calendar effects which remain available.
EXCLUDE_PREFIXES = (
    "merchant__new_observed_customers__",
    "merchant__returning_observed_customers__",
)
EXCLUDE_EXACT = {
    "merchant__new_customer_share",
    "merchant__returning_customer_share",
    "calendar__year_sin",
    "calendar__year_cos",
}


def filter_comparable(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep = [
        c for c in X.columns
        if c not in EXCLUDE_EXACT and not c.startswith(EXCLUDE_PREFIXES)
    ]
    return X[keep].copy(), keep


def bootstrap_ci(y, score, threshold, n_boot=3000):
    rng = np.random.default_rng(SEED)
    y = np.asarray(y, int); score = np.asarray(score, float); n = len(y)
    vals = {"roc_auc": [], "pr_auc": [], "precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n); yy = y[idx]; ss = score[idx]
        if len(np.unique(yy)) < 2:
            continue
        m = v71.metrics(yy, ss, threshold)
        vals["roc_auc"].append(float(roc_auc_score(yy, ss)))
        vals["pr_auc"].append(float(average_precision_score(yy, ss)))
        vals["precision"].append(m["precision"]); vals["recall"].append(m["recall"]); vals["f1"].append(m["f1"])
    return {k: {"low": float(np.quantile(v, .025)), "high": float(np.quantile(v, .975)), "n": len(v)} for k, v in vals.items()}


def main():
    dev_daily = v16.source_daily()
    ext_daily = v16.redsea_daily()
    dev_meta, Xdev0, cols0 = v16.build_meta_and_features(dev_daily)
    ext_meta, Xext0, ext_cols0 = v16.build_meta_and_features(ext_daily)
    if cols0 != ext_cols0:
        raise RuntimeError("V16 source/external schema mismatch")

    Xdev, cols = filter_comparable(Xdev0)
    Xext, ext_cols = filter_comparable(Xext0)
    if cols != ext_cols or list(Xdev.columns) != list(Xext.columns):
        raise RuntimeError("V16.1 comparable feature schema mismatch")
    if len(dev_meta) != 541:
        raise RuntimeError(f"Expected 541 development rows, got {len(dev_meta)}")

    run, oof = v71.nested_run(dev_meta, Xdev, cols, "portable_comparable_merchant")
    choices = [("portable_comparable_merchant", name, result) for name, result in run["models"].items()]
    _, selected_model, selected_result = max(choices, key=v71.selection_key)
    threshold = float(selected_result["median_inner_threshold"])
    dev_metrics = selected_result["nested_oof_metrics"]
    selected_oof = oof[oof.model.eq(selected_model)].sort_values(["fold_id", "date"]).copy()
    selected_oof.to_csv(DEV_OOF, index=False)

    ext_score, prep, fitted_kind, Xfit, Xout = v16.fit_full(selected_model, Xdev, dev_meta.target.astype(int), Xext)
    yext = ext_meta.target.to_numpy(int)
    ext_metrics = v71.metrics(yext, ext_score, threshold)
    ext_pred = ext_meta.copy(); ext_pred["score"] = ext_score; ext_pred["prediction"] = (ext_score >= threshold).astype(int)
    ext_pred.to_csv(REDSEA_PRED, index=False)

    dr = v16.drift_table(Xfit, Xout, prep)
    dr.to_csv(DRIFT, index=False)

    removed = [c for c in cols0 if c not in cols]
    v16_report = json.loads((ROOT / "reports" / "redsea_portable_v16" / "diagnostic_report.json").read_text(encoding="utf-8"))
    report = {
        "version": VERSION,
        "status": "POST_OPEN_EXTERNAL_DIAGNOSTIC_COMPARABILITY_ABLATION",
        "scientific_boundary": (
            "Redsea was already opened before V16.1, so this is not a new blind validation. "
            "Feature exclusions are based on semantic non-comparability of observation-window-dependent customer-history features and year-cycle encodings, not on Redsea outcome labels. "
            "Model and threshold are still selected only from nested rolling-origin development data."
        ),
        "feature_ablation": {
            "v16_feature_count": len(cols0),
            "v16_1_feature_count": len(cols),
            "removed_count": len(removed),
            "removed_features": removed,
        },
        "development": {
            "rows": len(dev_meta), "positive_rate": float(dev_meta.target.mean()),
            "selected_model": selected_model, "threshold": threshold,
            "nested_oof_metrics": dev_metrics,
            "worst_fold_recall": selected_result["worst_fold_recall"],
            "max_fold_alert_rate": selected_result["max_fold_alert_rate"],
        },
        "redsea": {
            "eligible_rows": len(ext_meta), "positive_rate": float(ext_meta.target.mean()),
            "metrics": ext_metrics,
            "bootstrap_95pct_ci": bootstrap_ci(yext, ext_score, threshold),
        },
        "drift": {
            "median_abs_smd": float(dr.abs_smd.median()),
            "max_abs_smd": float(dr.abs_smd.max()),
            "features_abs_smd_ge_1": int((dr.abs_smd >= 1.0).sum()),
            "top_10": dr.head(10).to_dict("records"),
        },
        "comparison_to_v16": {
            "development_auc_delta": float(dev_metrics["roc_auc"] - v16_report["development"]["nested_oof_metrics"]["roc_auc"]),
            "development_f1_delta": float(dev_metrics["f1"] - v16_report["development"]["nested_oof_metrics"]["f1"]),
            "redsea_auc_delta": float(ext_metrics["roc_auc"] - v16_report["redsea"]["metrics_at_development_frozen_threshold"]["roc_auc"]),
            "redsea_recall_delta": float(ext_metrics["recall"] - v16_report["redsea"]["metrics_at_development_frozen_threshold"]["recall"]),
            "redsea_precision_delta": float(ext_metrics["precision"] - v16_report["redsea"]["metrics_at_development_frozen_threshold"]["precision"]),
        },
        "red_supported": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    dm = report["development"]; em = report["redsea"]["metrics"]; cp = report["comparison_to_v16"]; drift = report["drift"]
    lines = [
        "# Sales Sentinel V16.1 — Comparability Ablation", "",
        f"- Status: **{report['status']}**",
        f"- Features: **{len(cols0)} → {len(cols)}** (removed {len(removed)} semantically non-comparable features)",
        f"- Selected model: **{dm['selected_model']}**",
        f"- Frozen nested threshold: **{dm['threshold']:.3f}**", "",
        "## Development nested OOF",
        f"- ROC-AUC / PR-AUC: **{dm['nested_oof_metrics']['roc_auc']:.2%} / {dm['nested_oof_metrics']['pr_auc']:.2%}**",
        f"- Precision / Recall / F1: **{dm['nested_oof_metrics']['precision']:.2%} / {dm['nested_oof_metrics']['recall']:.2%} / {dm['nested_oof_metrics']['f1']:.2%}**",
        f"- NPV / Alert rate: **{dm['nested_oof_metrics']['green_npv']:.2%} / {dm['nested_oof_metrics']['alert_rate']:.2%}**",
        f"- Δ AUC / Δ F1 vs V16: **{cp['development_auc_delta']:+.2%} / {cp['development_f1_delta']:+.2%}**", "",
        "## Redsea post-open diagnostic",
        f"- ROC-AUC / PR-AUC: **{em['roc_auc']:.2%} / {em['pr_auc']:.2%}**",
        f"- Precision / Recall / F1: **{em['precision']:.2%} / {em['recall']:.2%} / {em['f1']:.2%}**",
        f"- Accuracy / Balanced Accuracy: **{em['accuracy']:.2%} / {em['balanced_accuracy']:.2%}**",
        f"- NPV / Alert rate: **{em['green_npv']:.2%} / {em['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{em['tp']}/{em['fp']}/{em['fn']}/{em['tn']}**",
        f"- Δ external AUC / Recall / Precision vs V16: **{cp['redsea_auc_delta']:+.2%} / {cp['redsea_recall_delta']:+.2%} / {cp['redsea_precision_delta']:+.2%}**", "",
        "## Drift after comparability ablation",
        f"- Median |SMD|: **{drift['median_abs_smd']:.3f}**",
        f"- Max |SMD|: **{drift['max_abs_smd']:.3f}**",
        f"- Features |SMD|>=1: **{drift['features_abs_smd_ge_1']}**", "",
        "Scientific note: this is a post-open diagnostic, not a new blind validation. The ablation does not use Redsea labels to choose the model, threshold, or removed features.",
    ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
