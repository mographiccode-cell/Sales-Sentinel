from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "saudi_v1_3" / "saudi_daily_sama_calibrated_v1_3.csv"
OUT_DIR = ROOT / "models" / "saudi_v1_4"
REPORT_DIR = ROOT / "reports" / "saudi_v1_4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 7
DECLINE_THRESHOLD = 0.20
BASELINE_WINDOW = 28
TEST_DAYS = 90
PURGE = HORIZON
MIN_HISTORY = 56

MODEL_PATH = OUT_DIR / "sales_decline_classifier_v1_4.joblib"
META_PATH = OUT_DIR / "model_metadata_v1_4.json"
REPORT_PATH = REPORT_DIR / "sales_decline_retraining_report_v1_4.json"
SUMMARY_PATH = REPORT_DIR / "sales_decline_retraining_summary_v1_4.md"


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0, np.nan)


def build_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").reset_index(drop=True)

    sales = d["sama_calibrated_net_sales_sar"].astype(float)
    customers = d["unique_observed_customers"].astype(float)
    invoices = d["invoice_count"].astype(float)
    transactions = d["transaction_rows"].astype(float)
    new_customers = d["new_observed_customers"].astype(float)
    returning = d["returning_observed_customers"].astype(float)
    avg_invoice = d["average_invoice_value_sar"].astype(float)
    return_rate = d["return_rate_value"].astype(float)
    market_index = d["sama_weekly_market_index"].astype(float)

    x = pd.DataFrame({"date": d["date"]})
    # At prediction origin t, all same-day merchant observations are known.
    current_series = {
        "sales": sales,
        "customers": customers,
        "invoices": invoices,
        "transactions": transactions,
        "new_customers": new_customers,
        "returning_customers": returning,
        "avg_invoice": avg_invoice,
        "return_rate": return_rate,
    }
    for name, s in current_series.items():
        x[f"{name}_t0"] = s
        for lag in (1, 2, 3, 7, 14, 21, 28, 35, 56):
            x[f"{name}_lag_{lag}"] = s.shift(lag)
        for window in (7, 14, 28, 56):
            roll = s.rolling(window)
            x[f"{name}_mean_{window}"] = roll.mean()
            x[f"{name}_std_{window}"] = roll.std()
            x[f"{name}_median_{window}"] = roll.median()

    # Momentum/ratio features use only information available at t.
    x["sales_t0_vs_mean7"] = _safe_div(x["sales_t0"], x["sales_mean_7"])
    x["sales_t0_vs_mean28"] = _safe_div(x["sales_t0"], x["sales_mean_28"])
    x["sales_mean7_vs_mean28"] = _safe_div(x["sales_mean_7"], x["sales_mean_28"])
    x["customers_t0_vs_mean7"] = _safe_div(x["customers_t0"], x["customers_mean_7"])
    x["customers_mean7_vs_mean28"] = _safe_div(x["customers_mean_7"], x["customers_mean_28"])
    x["invoices_t0_vs_mean28"] = _safe_div(x["invoices_t0"], x["invoices_mean_28"])
    x["transactions_t0_vs_mean28"] = _safe_div(x["transactions_t0"], x["transactions_mean_28"])
    x["returning_share_t0"] = _safe_div(x["returning_customers_t0"], x["customers_t0"])
    x["new_customer_share_t0"] = _safe_div(x["new_customers_t0"], x["customers_t0"])

    # SAMA is an external aggregate calibration source. Only fully lagged prior-week values
    # are allowed as predictive features; the current/future SAMA week is never used.
    x["sama_market_index_lag_7"] = market_index.shift(7)
    x["sama_market_index_lag_14"] = market_index.shift(14)
    x["sama_market_index_lag_28"] = market_index.shift(28)

    baseline = sales.rolling(BASELINE_WINDOW).mean()
    future_mean = pd.concat([sales.shift(-i) for i in range(1, HORIZON + 1)], axis=1).mean(axis=1)
    x["baseline_sales_28"] = baseline
    x["future_sales_mean_7"] = future_mean
    x["future_decline_pct"] = 1.0 - _safe_div(future_mean, baseline)
    x["target_sales_decline_7d_20pct"] = (future_mean < (1.0 - DECLINE_THRESHOLD) * baseline).astype(float)

    # The final HORIZON origins have incomplete future windows and must never become labels.
    x.loc[len(x) - HORIZON :, ["future_sales_mean_7", "future_decline_pct", "target_sales_decline_7d_20pct"]] = np.nan
    x = x.iloc[MIN_HISTORY:].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    forbidden = {
        "date", "future_sales_mean_7", "future_decline_pct", "target_sales_decline_7d_20pct",
    }
    features = [c for c in x.columns if c not in forbidden]
    # baseline is known at origin t and is therefore allowed as a feature.
    assert "sama_weekly_market_index" not in features
    assert "future_sales_mean_7" not in features
    return x, features


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "Accuracy": float(accuracy_score(y_true, pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else None,
        "ConfusionMatrix": cm.tolist(),
    }


def threshold_score(m: dict) -> float:
    # Do not optimize raw accuracy alone on an imbalanced target.
    return 0.45 * m["BalancedAccuracy"] + 0.35 * m["F1"] + 0.20 * m["Accuracy"]


def model_specs() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", C=0.5, random_state=SEED)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=800, max_depth=8, min_samples_leaf=4,
            max_features="sqrt", class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=1000, max_depth=10, min_samples_leaf=3,
            max_features=0.7, class_weight="balanced", random_state=SEED, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.035, max_iter=350, max_leaf_nodes=15,
            min_samples_leaf=12, l2_regularization=2.0, random_state=SEED,
        ),
    }


def get_prob(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    score = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-score))


def main() -> None:
    daily = pd.read_csv(DATA)
    frame, features = build_supervised(daily)
    y = frame["target_sales_decline_7d_20pct"].astype(int).to_numpy()

    if len(frame) < 400:
        raise RuntimeError(f"Too few supervised origins: {len(frame)}")
    test_start = len(frame) - TEST_DAYS
    if test_start <= 260:
        raise RuntimeError("Not enough pre-test history")

    # Test remains untouched during model/threshold selection.
    test = frame.iloc[test_start:].copy()
    pretest_end = test_start - PURGE
    pretest = frame.iloc[:pretest_end].copy()

    # Three expanding walk-forward folds ending before the final test, each with purge.
    val_size = 60
    fold_starts = [pretest_end - 3 * val_size, pretest_end - 2 * val_size, pretest_end - val_size]
    folds = []
    min_train = 220
    for val_start in fold_starts:
        val_end = val_start + val_size
        train_end = val_start - PURGE
        if train_end < min_train:
            continue
        folds.append((0, train_end, val_start, val_end))
    if len(folds) < 3:
        raise RuntimeError("Could not construct three leakage-safe walk-forward folds")

    selection = {}
    oof_predictions = {}
    for name, spec in model_specs().items():
        fold_probs = []
        fold_true = []
        fold_rows = []
        for fold_no, (_, train_end, val_start, val_end) in enumerate(folds, 1):
            train = frame.iloc[:train_end]
            val = frame.iloc[val_start:val_end]
            y_train = train["target_sales_decline_7d_20pct"].astype(int).to_numpy()
            y_val = val["target_sales_decline_7d_20pct"].astype(int).to_numpy()
            if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
                raise RuntimeError(f"Fold {fold_no} has a single target class")
            model = clone(spec).fit(train[features], y_train)
            prob = get_prob(model, val[features])
            fold_probs.extend(prob.tolist())
            fold_true.extend(y_val.tolist())
            fold_rows.append({
                "fold": fold_no,
                "train_end_date": str(train["date"].iloc[-1].date()),
                "validation_start_date": str(val["date"].iloc[0].date()),
                "validation_end_date": str(val["date"].iloc[-1].date()),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_val.mean()),
            })

        fold_probs_arr = np.asarray(fold_probs)
        fold_true_arr = np.asarray(fold_true)
        best = None
        for threshold in np.arange(0.10, 0.901, 0.005):
            m = metrics(fold_true_arr, fold_probs_arr, float(threshold))
            score = threshold_score(m)
            candidate = (score, m["BalancedAccuracy"], m["F1"], -abs(threshold - 0.5), float(threshold), m)
            if best is None or candidate[:4] > best[:4]:
                best = candidate
        assert best is not None
        selection[name] = {
            "OOFThreshold": best[4],
            "OOFMetrics": best[5],
            "SelectionScore": best[0],
            "Folds": fold_rows,
        }
        oof_predictions[name] = {"y": fold_true, "prob": fold_probs}

    selected_name = max(
        selection,
        key=lambda n: (
            selection[n]["SelectionScore"],
            selection[n]["OOFMetrics"]["BalancedAccuracy"],
            selection[n]["OOFMetrics"]["F1"],
        ),
    )
    selected_threshold = float(selection[selected_name]["OOFThreshold"])

    # Final fit uses all origins whose 7-day target window ends before final test begins.
    fit_frame = frame.iloc[:pretest_end]
    final_model = clone(model_specs()[selected_name]).fit(
        fit_frame[features], fit_frame["target_sales_decline_7d_20pct"].astype(int)
    )
    test_y = test["target_sales_decline_7d_20pct"].astype(int).to_numpy()
    test_prob = get_prob(final_model, test[features])
    test_metrics = metrics(test_y, test_prob, selected_threshold)

    # Baselines for context, not for selection.
    always_no_prob = np.zeros(len(test_y))
    baseline_metrics = metrics(test_y, always_no_prob, 0.5)
    prior_rate = float(fit_frame["target_sales_decline_7d_20pct"].mean())

    acceptance = {
        "accuracy_at_least_90pct": test_metrics["Accuracy"] >= 0.90,
        "balanced_accuracy_at_least_80pct": test_metrics["BalancedAccuracy"] >= 0.80,
        "f1_at_least_70pct": test_metrics["F1"] >= 0.70,
        "roc_auc_at_least_85pct": (test_metrics["ROC_AUC"] or 0.0) >= 0.85,
        "beats_majority_accuracy": test_metrics["Accuracy"] > baseline_metrics["Accuracy"],
    }

    artifact = {
        "version": "SALES-DECLINE-1.4-LEAKAGE-SAFE",
        "dataset_version": "SA-LOCALIZATION-1.3.1-SAMA-SAFE",
        "problem_found": {
            "previous_target": "same-day unique-customer decline versus trailing 28-day customer mean",
            "previous_test_accuracy": 0.75,
            "previous_validation_dummy_accuracy": 0.80,
            "diagnosis": "Raw accuracy was misleading under imbalance, and the target did not match the Sales Sentinel core objective of predicting sales decline.",
        },
        "new_target": {
            "definition": "At prediction origin t, flag decline when mean SAMA-calibrated merchant sales over t+1..t+7 is at least 20% below the trailing 28-day mean known at t.",
            "horizon_days": HORIZON,
            "decline_threshold": DECLINE_THRESHOLD,
            "baseline_days": BASELINE_WINDOW,
        },
        "data": {
            "daily_rows": int(len(daily)),
            "supervised_origins": int(len(frame)),
            "pretest_fit_origins": int(len(fit_frame)),
            "test_origins": int(len(test)),
            "fit_positive_rate": prior_rate,
            "test_positive_rate": float(test_y.mean()),
            "test_start_date": str(test["date"].iloc[0].date()),
            "test_end_date": str(test["date"].iloc[-1].date()),
            "purge_days": PURGE,
        },
        "features": features,
        "selection_walk_forward": selection,
        "selected_model": selected_name,
        "selected_probability_threshold": selected_threshold,
        "final_test": test_metrics,
        "majority_baseline_test": baseline_metrics,
        "acceptance": acceptance,
        "all_acceptance_gates_passed": bool(all(acceptance.values())),
        "libraries": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "leakage_controls": {
            "shuffle": False,
            "final_test_used_for_selection": False,
            "purged_gap_equals_forecast_horizon": True,
            "future_sama_values_used_as_features": False,
            "current_same_week_sama_index_used_as_feature": False,
            "only_lagged_sama_index_7_14_28_days": True,
        },
    }

    joblib.dump({
        "model": final_model,
        "features": features,
        "threshold": selected_threshold,
        "target_horizon_days": HORIZON,
        "decline_threshold": DECLINE_THRESHOLD,
        "baseline_window_days": BASELINE_WINDOW,
        "version": artifact["version"],
    }, MODEL_PATH)
    META_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    m = test_metrics
    summary = f"""# Sales Decline Retraining v1.4\n\n- Dataset: **SA-LOCALIZATION-1.3.1-SAMA-SAFE**\n- Target: next **7-day mean sales** falls by at least **20%** versus trailing **28-day mean**.\n- Final test is the last **{len(test)}** prediction origins and was untouched during selection.\n- Purge gap: **{PURGE} days**.\n- Selected model: **{selected_name}**\n- Selected probability threshold: **{selected_threshold:.3f}**\n- Test Accuracy: **{m['Accuracy']:.3%}**\n- Test Balanced Accuracy: **{m['BalancedAccuracy']:.3%}**\n- Test Precision: **{m['Precision']:.3%}**\n- Test Recall: **{m['Recall']:.3%}**\n- Test F1: **{m['F1']:.3%}**\n- Test ROC-AUC: **{m['ROC_AUC']:.3%}**\n- Majority baseline Accuracy: **{baseline_metrics['Accuracy']:.3%}**\n- All acceptance gates passed: **{all(acceptance.values())}**\n\nThe final test was not used to choose the model or threshold.\n"""
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(json.dumps({
        "selected_model": selected_name,
        "threshold": selected_threshold,
        "test": test_metrics,
        "baseline": baseline_metrics,
        "acceptance": acceptance,
        "all_acceptance_gates_passed": all(acceptance.values()),
    }, indent=2))


if __name__ == "__main__":
    main()
