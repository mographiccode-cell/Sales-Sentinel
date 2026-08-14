from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "saudi_v1_3" / "saudi_daily_sama_calibrated_v1_3.csv"
DIAGNOSIS = ROOT / "reports" / "saudi_v1_5" / "target_diagnosis_preholdout.json"
MODEL_DIR = ROOT / "models" / "saudi_v1_5"
REPORT_DIR = ROOT / "reports" / "saudi_v1_5"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 7
DECLINE = 0.10
BASELINE = 28
DEV_FRACTION = 0.75
PURGE = HORIZON
MIN_HISTORY = 56

RAMADAN = [("2023-03-23", "2023-04-20"), ("2024-03-11", "2024-04-09")]
EID_FITR = [("2023-04-21", "2023-04-23"), ("2024-04-10", "2024-04-12")]
HAJJ = [("2023-06-19", "2023-06-30"), ("2024-06-07", "2024-06-19")]
EID_ADHA = [("2023-06-28", "2023-07-01"), ("2024-06-16", "2024-06-19")]


def in_ranges(date: pd.Timestamp, ranges) -> int:
    return int(any(pd.Timestamp(a) <= date <= pd.Timestamp(b) for a, b in ranges))


def safe_ratio(a, b):
    return a / b.replace(0, np.nan)


def build_frame(daily: pd.DataFrame):
    d = daily.copy().sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    sales = d["sama_calibrated_net_sales_sar"].astype(float)
    baseline = sales.rolling(BASELINE).mean()
    future = pd.concat([sales.shift(-i) for i in range(1, HORIZON + 1)], axis=1).mean(axis=1)

    x = pd.DataFrame({"date": d["date"]})
    core = {
        "sales": sales,
        "customers": d["unique_observed_customers"].astype(float),
        "invoices": d["invoice_count"].astype(float),
        "transactions": d["transaction_rows"].astype(float),
        "avg_invoice": d["average_invoice_value_sar"].astype(float),
        "return_rate": d["return_rate_value"].astype(float),
        "returning": d["returning_observed_customers"].astype(float),
        "new_customers": d["new_observed_customers"].astype(float),
    }
    for name, s in core.items():
        x[f"{name}_t0"] = s
        for lag in (1, 2, 3, 7, 14, 28, 56):
            x[f"{name}_lag_{lag}"] = s.shift(lag)
        for w in (7, 14, 28, 56):
            x[f"{name}_mean_{w}"] = s.rolling(w).mean()
            x[f"{name}_std_{w}"] = s.rolling(w).std()

    x["sales_vs_mean7"] = safe_ratio(x["sales_t0"], x["sales_mean_7"])
    x["sales_vs_mean28"] = safe_ratio(x["sales_t0"], x["sales_mean_28"])
    x["sales_mean7_vs_28"] = safe_ratio(x["sales_mean_7"], x["sales_mean_28"])
    x["sales_mean14_vs_28"] = safe_ratio(x["sales_mean_14"], x["sales_mean_28"])
    x["customers_vs_mean28"] = safe_ratio(x["customers_t0"], x["customers_mean_28"])
    x["invoices_vs_mean28"] = safe_ratio(x["invoices_t0"], x["invoices_mean_28"])
    x["transactions_vs_mean28"] = safe_ratio(x["transactions_t0"], x["transactions_mean_28"])
    x["returning_share"] = safe_ratio(x["returning_t0"], x["customers_t0"])
    x["new_customer_share"] = safe_ratio(x["new_customers_t0"], x["customers_t0"])

    market = d["sama_weekly_market_index"].astype(float)
    for lag in (7, 14, 21, 28):
        x[f"sama_index_lag_{lag}"] = market.shift(lag)
    x["sama_index_lag7_vs_28"] = safe_ratio(x["sama_index_lag_7"], x["sama_index_lag_28"])

    # Calendar features are known in advance and therefore valid forecast features.
    doy = d["date"].dt.dayofyear.astype(float)
    x["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    x["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    month = d["date"].dt.month.astype(float)
    x["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    x["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    x["is_national_day"] = ((d["date"].dt.month == 9) & (d["date"].dt.day == 23)).astype(int)
    x["is_founding_day"] = ((d["date"].dt.month == 2) & (d["date"].dt.day == 22)).astype(int)

    # Number of known Saudi-season days in the NEXT 7-day forecast window.
    future_ramadan, future_fitr, future_hajj, future_adha, future_national = [], [], [], [], []
    for origin in d["date"]:
        dates = [origin + pd.Timedelta(days=i) for i in range(1, HORIZON + 1)]
        future_ramadan.append(sum(in_ranges(dt, RAMADAN) for dt in dates))
        future_fitr.append(sum(in_ranges(dt, EID_FITR) for dt in dates))
        future_hajj.append(sum(in_ranges(dt, HAJJ) for dt in dates))
        future_adha.append(sum(in_ranges(dt, EID_ADHA) for dt in dates))
        future_national.append(sum(int(dt.month == 9 and dt.day == 23) for dt in dates))
    x["next7_ramadan_days"] = future_ramadan
    x["next7_eid_fitr_days"] = future_fitr
    x["next7_hajj_days"] = future_hajj
    x["next7_eid_adha_days"] = future_adha
    x["next7_national_day_count"] = future_national

    x["baseline_sales_28"] = baseline
    x["future_sales_mean_7"] = future
    x["future_decline_pct"] = 1 - safe_ratio(future, baseline)
    x["target"] = (future < (1 - DECLINE) * baseline).astype(float)
    x.loc[len(x) - HORIZON :, ["future_sales_mean_7", "future_decline_pct", "target"]] = np.nan
    x = x.iloc[MIN_HISTORY:].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    forbidden = {"date", "future_sales_mean_7", "future_decline_pct", "target"}
    features = [c for c in x.columns if c not in forbidden]
    return x, features


def get_prob(model, X):
    return model.predict_proba(X)[:, 1]


def eval_metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "Accuracy": float(accuracy_score(y, pred)),
        "BalancedAccuracy": float(balanced_accuracy_score(y, pred)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(y, p)),
        "ConfusionMatrix": cm.tolist(),
    }


def candidate_models():
    return {
        "LogisticRegression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.25, max_iter=5000, class_weight="balanced", random_state=SEED)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=1000, max_depth=7, min_samples_leaf=5, max_features=0.65,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=1200, max_depth=9, min_samples_leaf=4, max_features=0.65,
            class_weight="balanced", random_state=SEED, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.03, max_iter=400, max_leaf_nodes=15, min_samples_leaf=15,
            l2_regularization=3.0, random_state=SEED,
        ),
    }


def score(m):
    return 0.35*m["BalancedAccuracy"] + 0.30*m["F1"] + 0.20*m["Accuracy"] + 0.15*m["ROC_AUC"]


def main():
    diagnosis = json.loads(DIAGNOSIS.read_text(encoding="utf-8"))
    rec = diagnosis.get("recommended_target")
    if not rec or rec["horizon_days"] != HORIZON or abs(rec["decline_threshold"] - DECLINE) > 1e-9:
        raise RuntimeError("Pre-holdout diagnosis did not recommend the configured v1.5 target")

    daily = pd.read_csv(DATA, parse_dates=["date"])
    frame, features = build_frame(daily)
    raw_cutoff_date = daily.sort_values("date").iloc[int(len(daily)*DEV_FRACTION)-1]["date"]
    holdout_start_candidates = frame.index[frame["date"] > raw_cutoff_date].tolist()
    if not holdout_start_candidates:
        raise RuntimeError("No holdout after development cutoff")
    holdout_start = holdout_start_candidates[0]
    fit_end = holdout_start - PURGE
    dev = frame.iloc[:fit_end].copy()
    holdout = frame.iloc[holdout_start:].copy()
    if len(dev) < 300 or len(holdout) < 100:
        raise RuntimeError(f"Insufficient split dev={len(dev)} holdout={len(holdout)}")

    # 3 expanding walk-forward folds entirely inside the development period.
    val_size = 60
    starts = [len(dev)-180, len(dev)-120, len(dev)-60]
    folds = []
    for s in starts:
        train_end = s - PURGE
        if train_end < 180:
            raise RuntimeError("Insufficient training history for development fold")
        folds.append((train_end, s, s+val_size))

    selection = {}
    for name, spec in candidate_models().items():
        probs, truths, fold_meta = [], [], []
        for i, (train_end, val_start, val_end) in enumerate(folds, 1):
            tr = dev.iloc[:train_end]
            va = dev.iloc[val_start:val_end]
            ytr = tr["target"].astype(int).to_numpy()
            yva = va["target"].astype(int).to_numpy()
            model = clone(spec).fit(tr[features], ytr)
            p = get_prob(model, va[features])
            probs.extend(p.tolist()); truths.extend(yva.tolist())
            fold_meta.append({
                "fold": i,
                "train_end": str(tr["date"].iloc[-1].date()),
                "val_start": str(va["date"].iloc[0].date()),
                "val_end": str(va["date"].iloc[-1].date()),
                "val_positive_rate": float(yva.mean()),
            })
        probs = np.asarray(probs); truths = np.asarray(truths)
        best = None
        for threshold in np.arange(0.10, 0.901, 0.005):
            m = eval_metrics(truths, probs, float(threshold))
            # Require useful recall during development; otherwise high accuracy can be majority-class behavior.
            if m["Recall"] < 0.70:
                continue
            candidate = (score(m), m["BalancedAccuracy"], m["F1"], m["Accuracy"], -abs(threshold-0.5), float(threshold), m)
            if best is None or candidate[:5] > best[:5]:
                best = candidate
        if best is None:
            # If no threshold reaches recall 70%, retain the best balanced threshold but flag it.
            for threshold in np.arange(0.05, 0.951, 0.005):
                m = eval_metrics(truths, probs, float(threshold))
                candidate = (score(m), m["BalancedAccuracy"], m["F1"], m["Accuracy"], -abs(threshold-0.5), float(threshold), m)
                if best is None or candidate[:5] > best[:5]:
                    best = candidate
        selection[name] = {
            "threshold": best[5],
            "development_oof_metrics": best[6],
            "selection_score": best[0],
            "folds": fold_meta,
        }

    selected = max(selection, key=lambda n: (selection[n]["selection_score"], selection[n]["development_oof_metrics"]["BalancedAccuracy"], selection[n]["development_oof_metrics"]["F1"]))
    threshold = float(selection[selected]["threshold"])

    final_model = clone(candidate_models()[selected]).fit(dev[features], dev["target"].astype(int))
    y_hold = holdout["target"].astype(int).to_numpy()
    p_hold = get_prob(final_model, holdout[features])
    hold_metrics = eval_metrics(y_hold, p_hold, threshold)
    majority_accuracy = float(max(y_hold.mean(), 1-y_hold.mean()))

    gates = {
        "accuracy_at_least_90pct": hold_metrics["Accuracy"] >= 0.90,
        "balanced_accuracy_at_least_80pct": hold_metrics["BalancedAccuracy"] >= 0.80,
        "recall_at_least_75pct": hold_metrics["Recall"] >= 0.75,
        "f1_at_least_70pct": hold_metrics["F1"] >= 0.70,
        "roc_auc_at_least_85pct": hold_metrics["ROC_AUC"] >= 0.85,
        "beats_majority_accuracy": hold_metrics["Accuracy"] > majority_accuracy,
    }

    artifact = {
        "version": "SALES-DECLINE-1.5-STABLE-TARGET",
        "dataset_version": "SA-LOCALIZATION-1.3.1-SAMA-SAFE",
        "target_selection": {
            "source": "development-only pre-holdout diagnosis",
            "diagnosis_file": str(DIAGNOSIS.relative_to(ROOT)),
            "holdout_used_to_choose_target": False,
            "definition": "mean next 7-day SAMA-calibrated sales is at least 10% below trailing 28-day mean",
            "development_positive_rate_in_diagnosis": rec["positive_rate"],
            "development_segment_rates": rec["segment_positive_rates"],
        },
        "split": {
            "development_cutoff_date": str(raw_cutoff_date.date()),
            "development_origins_after_purge": int(len(dev)),
            "holdout_origins": int(len(holdout)),
            "holdout_start": str(holdout["date"].iloc[0].date()),
            "holdout_end": str(holdout["date"].iloc[-1].date()),
            "purge_days": PURGE,
            "shuffle": False,
        },
        "features": features,
        "model_selection": selection,
        "selected_model": selected,
        "selected_threshold": threshold,
        "holdout_positive_rate": float(y_hold.mean()),
        "holdout_metrics": hold_metrics,
        "holdout_majority_accuracy": majority_accuracy,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": bool(all(gates.values())),
        "leakage_controls": {
            "future_sales_in_features": False,
            "same_or_future_week_sama_index_in_features": False,
            "sama_index_minimum_lag_days": 7,
            "future_calendar_features_are_known_in_advance": True,
            "model_and_threshold_selected_before_holdout": True,
        },
        "libraries": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }

    joblib.dump({"model": final_model, "features": features, "threshold": threshold, "horizon_days": HORIZON, "decline_threshold": DECLINE, "baseline_days": BASELINE, "version": artifact["version"]}, MODEL_DIR / "sales_decline_classifier_v1_5.joblib")
    (MODEL_DIR / "model_metadata_v1_5.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (REPORT_DIR / "sales_decline_retraining_report_v1_5.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    m = hold_metrics
    (REPORT_DIR / "sales_decline_retraining_summary_v1_5.md").write_text(
        f"# Sales Decline v1.5\n\n- Target: next 7-day mean sales decline >= 10% vs trailing 28-day mean.\n- Selected model: **{selected}**\n- Threshold: **{threshold:.3f}**\n- Holdout: **{len(holdout)}** origins ({holdout['date'].iloc[0].date()} to {holdout['date'].iloc[-1].date()})\n- Positive rate: **{y_hold.mean():.2%}**\n- Accuracy: **{m['Accuracy']:.2%}**\n- Balanced Accuracy: **{m['BalancedAccuracy']:.2%}**\n- Precision: **{m['Precision']:.2%}**\n- Recall: **{m['Recall']:.2%}**\n- F1: **{m['F1']:.2%}**\n- ROC-AUC: **{m['ROC_AUC']:.2%}**\n- Majority baseline accuracy: **{majority_accuracy:.2%}**\n- All acceptance gates passed: **{all(gates.values())}**\n",
        encoding="utf-8",
    )
    print(json.dumps({"selected_model": selected, "threshold": threshold, "holdout_metrics": hold_metrics, "majority_accuracy": majority_accuracy, "gates": gates, "all_passed": all(gates.values())}, indent=2))

if __name__ == "__main__":
    main()
