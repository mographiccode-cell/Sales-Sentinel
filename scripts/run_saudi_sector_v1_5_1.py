from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

import build_train_saudi_panel_v1_4 as ml
import build_train_saudi_sector_panel_v1_5 as sector

VERSION = "SA-LOCALIZATION-1.5.1-SAMA-SECTOR-SAFE"
MIN_SPLIT_ROWS = 500
sector.VERSION = VERSION


def train_sector(d: pd.DataFrame, features: list[str], decline_threshold: float) -> dict:
    d = d.copy()
    d["target"] = (d["future_ratio"] < (1.0 - decline_threshold)).astype(int)

    train = d[d["TrainingSafeDate"] <= pd.Timestamp("2023-12-24")].copy()
    val = d[(d["TrainingSafeDate"] >= pd.Timestamp("2024-01-08")) & (d["TrainingSafeDate"] <= pd.Timestamp("2024-04-30"))].copy()
    test = d[d["TrainingSafeDate"] >= pd.Timestamp("2024-05-08")].copy()

    if min(len(train), len(val), len(test)) < MIN_SPLIT_ROWS:
        raise RuntimeError(f"Insufficient sector-panel splits: {len(train)}/{len(val)}/{len(test)}; need {MIN_SPLIT_ROWS} each")
    if min(train.target.nunique(), val.target.nunique(), test.target.nunique()) < 2:
        raise RuntimeError("One chronological split contains only one target class")
    for name, split in (("train", train), ("validation", val), ("test", test)):
        positives = int(split.target.sum())
        negatives = int(len(split) - positives)
        if min(positives, negatives) < 80:
            raise RuntimeError(f"{name} has too few examples of one class: positive={positives}, negative={negatives}")

    pos = int(train.target.sum())
    neg = len(train) - pos
    pos_weight = neg / max(pos, 1)

    cls_results = {}
    cls_scores = {}
    for name, model in ml.classifiers(pos_weight).items():
        fit = clone(model).fit(train[features], train.target)
        score = fit.predict_proba(val[features])[:, 1]
        threshold, metrics, selection_score = ml.best_threshold(val.target.to_numpy(), score)
        cls_results[name] = {"threshold": threshold, "metrics": metrics, "selection_score": selection_score}
        cls_scores[name] = score
    best_cls = max(cls_results, key=lambda n: (cls_results[n]["selection_score"], cls_results[n]["metrics"]["ROC_AUC"]))

    reg_results = {}
    reg_scores = {}
    for name, model in ml.regressors().items():
        fit = clone(model).fit(train[features], train["future_ratio"].clip(0.05, 3.0))
        ratio = fit.predict(val[features])
        risk = 1.0 / (1.0 + np.exp(np.clip((ratio - (1.0 - decline_threshold)) / 0.06, -30, 30)))
        threshold, metrics, selection_score = ml.best_threshold(val.target.to_numpy(), risk)
        reg_results[name] = {
            "threshold": threshold,
            "metrics": metrics,
            "selection_score": selection_score,
            "validation_ratio_mae": float(mean_absolute_error(val["future_ratio"], ratio)),
        }
        reg_scores[name] = risk
    best_reg = max(reg_results, key=lambda n: (reg_results[n]["selection_score"], reg_results[n]["metrics"]["ROC_AUC"]))

    # Blend and probability threshold are selected only on validation.
    best_blend = None
    for weight in np.linspace(0.0, 1.0, 21):
        score = weight * cls_scores[best_cls] + (1.0 - weight) * reg_scores[best_reg]
        threshold, metrics, selection_score = ml.best_threshold(val.target.to_numpy(), score)
        candidate = (
            selection_score,
            metrics["BalancedAccuracy"],
            metrics["F1"],
            metrics["ROC_AUC"],
            float(weight),
            threshold,
            metrics,
        )
        if best_blend is None or candidate[:4] > best_blend[:4]:
            best_blend = candidate
    blend_weight = best_blend[4]
    blend_threshold = best_blend[5]

    trainval = pd.concat([train, val], ignore_index=True).sort_values("TrainingSafeDate")
    tv_pos = int(trainval.target.sum())
    tv_neg = len(trainval) - tv_pos
    final_cls = clone(ml.classifiers(tv_neg / max(tv_pos, 1))[best_cls]).fit(trainval[features], trainval.target)
    final_reg = clone(ml.regressors()[best_reg]).fit(trainval[features], trainval["future_ratio"].clip(0.05, 3.0))

    test_cls = final_cls.predict_proba(test[features])[:, 1]
    test_ratio = final_reg.predict(test[features])
    test_reg_risk = 1.0 / (1.0 + np.exp(np.clip((test_ratio - (1.0 - decline_threshold)) / 0.06, -30, 30)))
    test_score = blend_weight * test_cls + (1.0 - blend_weight) * test_reg_risk
    test_metrics = ml.classification_metrics(test.target.to_numpy(), test_score, blend_threshold)
    majority = max(float(test.target.mean()), 1.0 - float(test.target.mean()))

    gates = {
        "accuracy_above_majority_by_5pp": test_metrics["Accuracy"] >= majority + 0.05,
        "balanced_accuracy_at_least_75pct": test_metrics["BalancedAccuracy"] >= 0.75,
        "recall_at_least_70pct": test_metrics["Recall"] >= 0.70,
        "f1_at_least_65pct": test_metrics["F1"] >= 0.65,
        "roc_auc_at_least_82pct": test_metrics["ROC_AUC"] >= 0.82,
    }

    joblib.dump({
        "classifier": final_cls,
        "regressor": final_reg,
        "features": features,
        "blend_weight_classifier": blend_weight,
        "blend_weight_regression": 1.0 - blend_weight,
        "probability_threshold": blend_threshold,
        "decline_threshold": decline_threshold,
        "horizon_days": sector.H,
        "baseline_days": sector.B,
        "version": VERSION,
        "classifier_name": best_cls,
        "regressor_name": best_reg,
    }, sector.MODEL)

    return {
        "split": {
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "minimum_required_rows_each": MIN_SPLIT_ROWS,
            "train_positive_count": int(train.target.sum()),
            "validation_positive_count": int(val.target.sum()),
            "test_positive_count": int(test.target.sum()),
            "train_end": "2023-12-24",
            "validation": "2024-01-08..2024-04-30",
            "test_start": "2024-05-08",
            "purge_days": 7,
            "shuffle": False,
        },
        "target": {
            "definition": f"mean sales of next {sector.H} calendar days is at least {decline_threshold:.1%} below trailing {sector.B}-day mean",
            "decline_threshold": decline_threshold,
            "train_positive_rate": float(train.target.mean()),
            "validation_positive_rate": float(val.target.mean()),
            "test_positive_rate": float(test.target.mean()),
        },
        "classification_candidates": cls_results,
        "regression_candidates": reg_results,
        "selected_classifier": best_cls,
        "selected_regressor": best_reg,
        "blend_weight_classifier": blend_weight,
        "blend_weight_regression": 1.0 - blend_weight,
        "selected_probability_threshold": blend_threshold,
        "test_metrics": test_metrics,
        "majority_test_accuracy": majority,
        "high_accuracy_90pct_goal_met": bool(test_metrics["Accuracy"] >= 0.90),
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": bool(all(gates.values())),
        "leakage_controls": {
            "actual_future_SAMA_values_used_as_features": False,
            "same_week_actual_SAMA_used_as_feature": False,
            "previous_completed_SAMA_week_only": True,
            "walk_forward_sector_SAMA_forecasts_only": True,
            "synthetic_region_used": False,
            "fallback_customer_ids_counted_as_customers": False,
            "future_target_columns_in_features": False,
            "chronological_split": True,
            "validation_only_model_blend_and_threshold_selection": True,
            "test_used_for_model_or_threshold_selection": False,
        },
    }


# Patch only the generic sample-size-specific training function; data preparation and all quality gates stay intact.
sector.ml.train_and_evaluate = train_sector

if __name__ == "__main__":
    sector.main()
