from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DAILY_FILE = ROOT / "data" / "saudi_v1_3" / "saudi_daily_sama_calibrated_v1_3.csv"
SAMA_FORECAST_FILE = ROOT / "data" / "sama_pos" / "sama_market_walkforward_forecasts_2023_2025.csv"
SAMA_REPORT_FILE = ROOT / "reports" / "sama_market_v1_6" / "sama_market_forecaster_report_v1_6.json"
TARGET_DIAGNOSIS = ROOT / "reports" / "saudi_v1_5" / "target_diagnosis_preholdout.json"
MODEL_DIR = ROOT / "models" / "saudi_v1_6"
REPORT_DIR = ROOT / "reports" / "saudi_v1_6"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 7
DECLINE = 0.10
BASELINE = 28
MIN_HISTORY = 56
PURGE = HORIZON
VAL_DAYS = 60
TEST_BLOCK_DAYS = 45
MIN_INNER_TRAIN = 220

RAMADAN = [("2023-03-23", "2023-04-20"), ("2024-03-11", "2024-04-09")]
EID_FITR = [("2023-04-21", "2023-04-23"), ("2024-04-10", "2024-04-12")]
HAJJ = [("2023-06-19", "2023-06-30"), ("2024-06-07", "2024-06-19")]
EID_ADHA = [("2023-06-28", "2023-07-01"), ("2024-06-16", "2024-06-19")]


def safe_ratio(a, b):
    return a / b.replace(0, np.nan)


def in_ranges(date: pd.Timestamp, ranges) -> int:
    return int(any(pd.Timestamp(a) <= date <= pd.Timestamp(b) for a, b in ranges))


def sama_week_start(date: pd.Timestamp) -> pd.Timestamp:
    # Sunday-Saturday week.
    return pd.Timestamp(date) - pd.Timedelta(days=(pd.Timestamp(date).dayofweek + 1) % 7)


def load_sama_forecast_lookup():
    report = json.loads(SAMA_REPORT_FILE.read_text(encoding="utf-8"))
    forecasts = pd.read_csv(SAMA_FORECAST_FILE, parse_dates=["origin_week_start", "forecast_h1_week_start", "forecast_h2_week_start"])
    lookup = forecasts.set_index("origin_week_start").to_dict("index")
    return lookup, report


def add_forecasted_market_features(x: pd.DataFrame, dates: pd.Series, lookup: dict):
    cols = {
        "forecast_sama_value_h1_index": [],
        "forecast_sama_value_h2_index": [],
        "forecast_sama_count_h1_index": [],
        "forecast_sama_count_h2_index": [],
        "forecast_sama_value_h1_change": [],
        "forecast_sama_value_h2_change": [],
        "forecast_sama_count_h1_change": [],
        "forecast_sama_count_h2_change": [],
        "forecast_sama_next7_weighted_value_index": [],
        "forecast_sama_next7_weighted_count_index": [],
    }
    for origin_date in dates:
        current_week = sama_week_start(pd.Timestamp(origin_date))
        last_completed = current_week - pd.Timedelta(days=7)
        row = lookup.get(last_completed)
        if row is None:
            for key in cols:
                cols[key].append(np.nan)
            continue
        h1_v = float(row["predicted_value_h1_index_52median"])
        h2_v = float(row["predicted_value_h2_index_52median"])
        h1_c = float(row["predicted_count_h1_index_52median"])
        h2_c = float(row["predicted_count_h2_index_52median"])
        future_week_starts = [sama_week_start(pd.Timestamp(origin_date) + pd.Timedelta(days=i)) for i in range(1, HORIZON+1)]
        h1_week = pd.Timestamp(row["forecast_h1_week_start"])
        h2_week = pd.Timestamp(row["forecast_h2_week_start"])
        value_indices, count_indices = [], []
        for wk in future_week_starts:
            if wk == h1_week:
                value_indices.append(h1_v); count_indices.append(h1_c)
            elif wk == h2_week:
                value_indices.append(h2_v); count_indices.append(h2_c)
            else:
                # Seven future days can touch at most h1/h2 when using last completed week.
                value_indices.append(h2_v); count_indices.append(h2_c)
        cols["forecast_sama_value_h1_index"].append(h1_v)
        cols["forecast_sama_value_h2_index"].append(h2_v)
        cols["forecast_sama_count_h1_index"].append(h1_c)
        cols["forecast_sama_count_h2_index"].append(h2_c)
        cols["forecast_sama_value_h1_change"].append(float(row["predicted_value_h1_change_vs_last"]))
        cols["forecast_sama_value_h2_change"].append(float(row["predicted_value_h2_change_vs_last"]))
        cols["forecast_sama_count_h1_change"].append(float(row["predicted_count_h1_change_vs_last"]))
        cols["forecast_sama_count_h2_change"].append(float(row["predicted_count_h2_change_vs_last"]))
        cols["forecast_sama_next7_weighted_value_index"].append(float(np.mean(value_indices)))
        cols["forecast_sama_next7_weighted_count_index"].append(float(np.mean(count_indices)))
    for key, values in cols.items():
        x[key] = values


def build_frame(daily: pd.DataFrame, sama_lookup: dict):
    d = daily.copy().sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"])
    sales = d["sama_calibrated_net_sales_sar"].astype(float)
    baseline = sales.rolling(BASELINE).mean()
    future = pd.concat([sales.shift(-i) for i in range(1, HORIZON+1)], axis=1).mean(axis=1)

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

    # Only genuinely lagged actual SAMA observations.
    market = d["sama_weekly_market_index"].astype(float)
    for lag in (7, 14, 21, 28):
        x[f"actual_sama_index_lag_{lag}"] = market.shift(lag)
    add_forecasted_market_features(x, d["date"], sama_lookup)

    doy = d["date"].dt.dayofyear.astype(float)
    x["doy_sin"] = np.sin(2*np.pi*doy/365.25); x["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    month = d["date"].dt.month.astype(float)
    x["month_sin"] = np.sin(2*np.pi*month/12.0); x["month_cos"] = np.cos(2*np.pi*month/12.0)
    next7 = [[dt + pd.Timedelta(days=i) for i in range(1, HORIZON+1)] for dt in d["date"]]
    x["next7_ramadan_days"] = [sum(in_ranges(dt, RAMADAN) for dt in ds) for ds in next7]
    x["next7_eid_fitr_days"] = [sum(in_ranges(dt, EID_FITR) for dt in ds) for ds in next7]
    x["next7_hajj_days"] = [sum(in_ranges(dt, HAJJ) for dt in ds) for ds in next7]
    x["next7_eid_adha_days"] = [sum(in_ranges(dt, EID_ADHA) for dt in ds) for ds in next7]
    x["next7_national_day_count"] = [sum(int(dt.month==9 and dt.day==23) for dt in ds) for ds in next7]

    x["baseline_sales_28"] = baseline
    x["future_sales_mean_7"] = future
    x["future_decline_pct"] = 1 - safe_ratio(future, baseline)
    x["target"] = (future < (1-DECLINE)*baseline).astype(float)
    x.loc[len(x)-HORIZON:, ["future_sales_mean_7", "future_decline_pct", "target"]] = np.nan
    x = x.iloc[MIN_HISTORY:].replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
    forbidden = {"date","future_sales_mean_7","future_decline_pct","target"}
    features = [c for c in x.columns if c not in forbidden]
    return x, features


def models():
    return {
        "LogisticRegression": Pipeline([("scale",StandardScaler()),("model",LogisticRegression(C=.25,max_iter=5000,class_weight="balanced",random_state=SEED))]),
        "RandomForest": RandomForestClassifier(n_estimators=800,max_depth=7,min_samples_leaf=5,max_features=.65,class_weight="balanced_subsample",random_state=SEED,n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=1000,max_depth=9,min_samples_leaf=4,max_features=.65,class_weight="balanced",random_state=SEED,n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(learning_rate=.03,max_iter=350,max_leaf_nodes=15,min_samples_leaf=15,l2_regularization=3,random_state=SEED),
    }


def metr(y,p,t):
    pred=(p>=t).astype(int); cm=confusion_matrix(y,pred,labels=[0,1])
    return {"Accuracy":float(accuracy_score(y,pred)),"BalancedAccuracy":float(balanced_accuracy_score(y,pred)),"Precision":float(precision_score(y,pred,zero_division=0)),"Recall":float(recall_score(y,pred,zero_division=0)),"F1":float(f1_score(y,pred,zero_division=0)),"ROC_AUC":float(roc_auc_score(y,p)),"ConfusionMatrix":cm.tolist()}


def rank_score(m):
    return .35*m["BalancedAccuracy"]+.30*m["F1"]+.20*m["Accuracy"]+.15*m["ROC_AUC"]


def choose_model_threshold(train: pd.DataFrame, features):
    val_start=len(train)-VAL_DAYS
    inner_end=val_start-PURGE
    if inner_end<MIN_INNER_TRAIN: raise RuntimeError("Insufficient inner training history")
    inner=train.iloc[:inner_end]; val=train.iloc[val_start:]
    result={}
    for name,spec in models().items():
        fit=clone(spec).fit(inner[features],inner["target"].astype(int))
        p=fit.predict_proba(val[features])[:,1]
        best=None
        for t in np.arange(.05,.951,.005):
            m=metr(val["target"].astype(int).to_numpy(),p,float(t))
            # Prefer useful sensitivity; no raw-accuracy shortcut.
            penalty=0 if m["Recall"]>=.70 else (.70-m["Recall"])
            tup=(rank_score(m)-.25*penalty,m["BalancedAccuracy"],m["F1"],m["Accuracy"],-abs(t-.5),float(t),m)
            if best is None or tup[:5]>best[:5]: best=tup
        result[name]={"threshold":best[5],"metrics":best[6],"score":best[0]}
    name=max(result,key=lambda n:(result[n]["score"],result[n]["metrics"]["BalancedAccuracy"],result[n]["metrics"]["F1"]))
    return name,float(result[name]["threshold"]),result


def main():
    diagnosis=json.loads(TARGET_DIAGNOSIS.read_text(encoding="utf-8"))
    rec=diagnosis["recommended_target"]
    if rec["horizon_days"]!=HORIZON or abs(rec["decline_threshold"]-DECLINE)>1e-9:
        raise RuntimeError("v1.6 target differs from development-only target diagnosis")
    sama_lookup,sama_report=load_sama_forecast_lookup()
    daily=pd.read_csv(DAILY_FILE,parse_dates=["date"])
    frame,features=build_frame(daily,sama_lookup)

    # Nested walk-forward evaluation. These are out-of-sample blocks; because prior experiments
    # exposed later-period metrics, we deliberately label this a backtest rather than an untouched test.
    first_test=max(MIN_INNER_TRAIN+VAL_DAYS+2*PURGE,260)
    starts=list(range(first_test,len(frame),TEST_BLOCK_DAYS))
    all_y=[]; all_p=[]; all_pred=[]; fold_reports=[]; selected_models=[]
    for fold_no,start in enumerate(starts,1):
        end=min(start+TEST_BLOCK_DAYS,len(frame))
        if end-start<20: continue
        fit_end=start-PURGE
        historical=frame.iloc[:fit_end].copy()
        test=frame.iloc[start:end].copy()
        name,threshold,selection=choose_model_threshold(historical,features)
        final=clone(models()[name]).fit(historical[features],historical["target"].astype(int))
        p=final.predict_proba(test[features])[:,1]
        y=test["target"].astype(int).to_numpy(); pred=(p>=threshold).astype(int)
        all_y.extend(y.tolist()); all_p.extend(p.tolist()); all_pred.extend(pred.tolist()); selected_models.append(name)
        fold_reports.append({"fold":fold_no,"train_end":str(historical["date"].iloc[-1].date()),"test_start":str(test["date"].iloc[0].date()),"test_end":str(test["date"].iloc[-1].date()),"selected_model":name,"threshold":threshold,"test_positive_rate":float(y.mean()),"metrics":metr(y,p,threshold),"selection":selection})
    y=np.asarray(all_y); p=np.asarray(all_p); pred=np.asarray(all_pred)
    if len(y)<100: raise RuntimeError("Insufficient nested walk-forward evaluation rows")
    cm=confusion_matrix(y,pred,labels=[0,1])
    aggregate={"Accuracy":float(accuracy_score(y,pred)),"BalancedAccuracy":float(balanced_accuracy_score(y,pred)),"Precision":float(precision_score(y,pred,zero_division=0)),"Recall":float(recall_score(y,pred,zero_division=0)),"F1":float(f1_score(y,pred,zero_division=0)),"ROC_AUC":float(roc_auc_score(y,p)),"ConfusionMatrix":cm.tolist(),"EvaluationRows":int(len(y)),"PositiveRate":float(y.mean())}
    majority=max(float(y.mean()),1-float(y.mean()))
    gates={"accuracy_at_least_90pct":aggregate["Accuracy"]>=.90,"balanced_accuracy_at_least_80pct":aggregate["BalancedAccuracy"]>=.80,"recall_at_least_75pct":aggregate["Recall"]>=.75,"f1_at_least_70pct":aggregate["F1"]>=.70,"roc_auc_at_least_85pct":aggregate["ROC_AUC"]>=.85,"beats_majority_accuracy":aggregate["Accuracy"]>majority}

    # Deployment model: choose most frequently selected class, tune threshold on latest prior validation.
    deployment_name=Counter(selected_models).most_common(1)[0][0]
    dep_train=frame.iloc[:-PURGE].copy()
    dep_name,dep_threshold,dep_selection=choose_model_threshold(dep_train,features)
    # Use the current selection if it agrees; otherwise use the latest validated selection, which is more current.
    deployment_name=dep_name
    deployment=clone(models()[deployment_name]).fit(dep_train[features],dep_train["target"].astype(int))
    joblib.dump({"model":deployment,"features":features,"threshold":dep_threshold,"target":{"horizon_days":HORIZON,"decline_threshold":DECLINE,"baseline_days":BASELINE},"version":"SALES-DECLINE-1.6-SAMA-FORECAST-AUGMENTED"},MODEL_DIR/"sales_decline_classifier_v1_6.joblib")

    report={"version":"SALES-DECLINE-1.6-SAMA-FORECAST-AUGMENTED","dataset":"SA-LOCALIZATION-1.3.1-SAMA-SAFE","target":"next 7-day mean sales >=10% below trailing 28-day mean","evaluation_type":"nested expanding walk-forward backtest; not described as untouched test because previous experiments exposed later-period metrics","sama_forecaster":sama_report,"forecast_feature_contract":"Every daily origin uses forecasts generated from the last completed SAMA week only; actual future/current incomplete SAMA values are never model features.","feature_count":len(features),"features":features,"folds":fold_reports,"aggregate_backtest":aggregate,"majority_accuracy":majority,"acceptance_gates":gates,"all_acceptance_gates_passed":bool(all(gates.values())),"deployment":{"selected_model":deployment_name,"threshold":dep_threshold,"latest_validation_selection":dep_selection}}
    (REPORT_DIR/"sales_decline_v1_6_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (MODEL_DIR/"model_metadata_v1_6.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    (REPORT_DIR/"sales_decline_v1_6_summary.md").write_text(f"# Sales Decline v1.6\n\n- Evaluation: nested walk-forward backtest\n- Rows: **{len(y)}**\n- Positive rate: **{y.mean():.2%}**\n- Accuracy: **{aggregate['Accuracy']:.2%}**\n- Balanced Accuracy: **{aggregate['BalancedAccuracy']:.2%}**\n- Precision: **{aggregate['Precision']:.2%}**\n- Recall: **{aggregate['Recall']:.2%}**\n- F1: **{aggregate['F1']:.2%}**\n- ROC-AUC: **{aggregate['ROC_AUC']:.2%}**\n- Majority baseline Accuracy: **{majority:.2%}**\n- Deployment model: **{deployment_name}**\n- All acceptance gates passed: **{all(gates.values())}**\n",encoding="utf-8")
    print(json.dumps({"aggregate_backtest":aggregate,"majority_accuracy":majority,"gates":gates,"all_passed":all(gates.values()),"deployment_model":deployment_name,"threshold":dep_threshold},indent=2))

if __name__=="__main__": main()
