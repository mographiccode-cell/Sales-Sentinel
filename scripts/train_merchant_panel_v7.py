from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "merchant_v4" / "merchant_sector_daily_features_v4.csv"
OUT = ROOT / "reports" / "merchant_panel_v7"
MOD = ROOT / "models" / "merchant_panel_v7"
DATA = ROOT / "data" / "merchant_v7"
for p in (OUT, MOD, DATA): p.mkdir(parents=True, exist_ok=True)
SEED = 42
DECLINE = 0.15
PURGE_DAYS = 7
BLIND_DAYS = 84


def metrics(y, p, t=0.5):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "accuracy": float(accuracy_score(y,pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y,pred)),
        "precision": float(precision_score(y,pred,zero_division=0)),
        "recall": float(recall_score(y,pred,zero_division=0)),
        "f1": float(f1_score(y,pred,zero_division=0)),
        "roc_auc": float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
        "pr_auc": float(average_precision_score(y,p)) if len(np.unique(y))>1 else None,
        "alert_rate": float(pred.mean()),
        "green_npv": float(tn/max(tn+fn,1)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }


def load_and_revalidate():
    d = pd.read_csv(SRC)
    required = {"date","category","future_ratio","target"}
    if not required.issubset(d.columns):
        raise RuntimeError(f"Missing required columns: {sorted(required-set(d.columns))}")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date","category","future_ratio"]).sort_values(["date","category"]).reset_index(drop=True)
    d = d.drop_duplicates(subset=["date","category"], keep="last")
    d["target_v7"] = (pd.to_numeric(d["future_ratio"], errors="coerce") < (1-DECLINE)).astype(int)
    # Features only: explicitly exclude labels, future values and post-outcome columns.
    forbidden_exact = {"target","target_v7","future_ratio","future7_sales","baseline28_daily","date","category","sama_sector"}
    forbidden_fragments = ("future", "target", "actual_")
    feature_cols=[]
    for c in d.columns:
        if c in forbidden_exact or any(x in c.lower() for x in forbidden_fragments): continue
        if pd.api.types.is_numeric_dtype(d[c]): feature_cols.append(c)
    if not feature_cols: raise RuntimeError("No numeric leakage-safe features found")
    X=d[feature_cols].replace([np.inf,-np.inf],np.nan).copy()
    # Robust clipping thresholds are fitted later on training data only.
    meta=d[["date","category","target_v7","future_ratio"]].copy()
    return meta,X,feature_cols


def factories():
    return {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=.2,class_weight="balanced",max_iter=4000,random_state=SEED)),
        "extra_trees": make_pipeline(SimpleImputer(strategy="median"), ExtraTreesClassifier(n_estimators=350,min_samples_leaf=5,max_features=.55,class_weight="balanced",n_jobs=-1,random_state=SEED)),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=350,min_samples_leaf=5,max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=SEED)),
        "hist_gb": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=250,learning_rate=.045,max_leaf_nodes=15,l2_regularization=2.0,random_state=SEED))
    }


def clip_train_only(Xtr,Xv):
    lo=Xtr.quantile(.005); hi=Xtr.quantile(.995)
    return Xtr.clip(lo,hi,axis=1), Xv.clip(lo,hi,axis=1), lo, hi


def choose_threshold(y,p):
    best=None
    for t in np.arange(.20,.801,.01):
        m=metrics(y,p,float(t))
        # Operational contract: prioritize recall and NPV, then F1/precision, penalize alert excess.
        feasible=(m["recall"]>=.78 and m["green_npv"]>=.94 and m["alert_rate"]<=.45)
        score=(1000 if feasible else 0)+2*m["f1"]+m["precision"]+m["recall"]-.7*max(0,m["alert_rate"]-.35)
        if best is None or score>best[0]: best=(score,float(t),m)
    return best[1],best[2]


def main():
    meta,X,features=load_and_revalidate()
    dates=np.array(sorted(meta.date.dt.normalize().unique()))
    blind_start=pd.Timestamp(dates[-BLIND_DAYS])
    dev_end=blind_start-pd.Timedelta(days=PURGE_DAYS+1)
    dev_dates=np.array([x for x in dates if pd.Timestamp(x)<=dev_end])
    # Four expanding rolling-origin folds over development period.
    anchors=np.linspace(int(len(dev_dates)*.45), int(len(dev_dates)*.82), 4).astype(int)
    val_len=max(35,int(len(dev_dates)*.12))
    oof=[]; fold_reports=[]
    models=factories()
    for fid,a in enumerate(anchors):
        vs=pd.Timestamp(dev_dates[a]); ve=pd.Timestamp(dev_dates[min(a+val_len-1,len(dev_dates)-1)])
        te=vs-pd.Timedelta(days=PURGE_DAYS+1)
        tr=meta.date<=te; va=(meta.date>=vs)&(meta.date<=ve)
        if tr.sum()<100 or va.sum()<40: continue
        Xtr,Xva,_,_=clip_train_only(X.loc[tr],X.loc[va]); ytr=meta.loc[tr,"target_v7"].to_numpy(); yva=meta.loc[va,"target_v7"].to_numpy()
        fold_probs={}
        for name,f in models.items():
            model=clone(f); model.fit(Xtr,ytr); fold_probs[name]=model.predict_proba(Xva)[:,1]
        for name,p in fold_probs.items():
            for i,(idx,prob) in enumerate(zip(meta.index[va],p)):
                oof.append({"row":int(idx),"fold":fid,"model":name,"prob":float(prob),"y":int(meta.loc[idx,"target_v7"]),"date":str(meta.loc[idx,"date"].date())})
        fold_reports.append({"fold":fid,"train_end":str(te.date()),"val_start":str(vs.date()),"val_end":str(ve.date()),"train_rows":int(tr.sum()),"val_rows":int(va.sum()),"positives":int(yva.sum())})
    od=pd.DataFrame(oof)
    if od.empty: raise RuntimeError("No OOF predictions generated")
    pivot=od.pivot_table(index=["row","y","date"],columns="model",values="prob").reset_index()
    y=pivot.y.to_numpy()
    candidates={name:pivot[name].to_numpy() for name in models}
    candidates["mean_ensemble"]=np.mean(np.column_stack([candidates[n] for n in models]),axis=1)
    dev_results={}
    best_name=None; best_key=None
    for name,p in candidates.items():
        t,m=choose_threshold(y,p); dev_results[name]={"threshold":t,"metrics":m}
        key=(m["f1"],m["pr_auc"] or 0,m["recall"],-m["alert_rate"])
        if best_key is None or key>best_key: best_key=key; best_name=name
    threshold=dev_results[best_name]["threshold"]

    # Final blind holdout is opened only after model+threshold selection.
    train_mask=meta.date<=dev_end; blind_mask=meta.date>=blind_start
    Xtr,Xb,lo,hi=clip_train_only(X.loc[train_mask],X.loc[blind_mask]); ytr=meta.loc[train_mask,"target_v7"].to_numpy(); yb=meta.loc[blind_mask,"target_v7"].to_numpy()
    fitted={}; probs=[]
    selected_members=list(models) if best_name=="mean_ensemble" else [best_name]
    for name in selected_members:
        model=clone(models[name]); model.fit(Xtr,ytr); fitted[name]=model; probs.append(model.predict_proba(Xb)[:,1])
    pb=np.mean(np.column_stack(probs),axis=1)
    blind=metrics(yb,pb,threshold)
    report={
      "version":"SALES-SENTINEL-V7-PANEL-BLIND-HOLDOUT",
      "scientific_status":"EXPERIMENTAL_EXTERNAL_SAUDI_VALIDATION_STILL_REQUIRED",
      "source_panel":str(SRC.relative_to(ROOT)),
      "panel_rows":int(len(meta)),"feature_count":len(features),"early_decline_threshold":DECLINE,
      "purge_days":PURGE_DAYS,"blind_holdout_days":BLIND_DAYS,"blind_start":str(blind_start.date()),
      "development_rows":int(train_mask.sum()),"blind_rows":int(blind_mask.sum()),"blind_positives":int(yb.sum()),
      "folds":fold_reports,"development_models":dev_results,"selected_model":best_name,"selected_threshold":threshold,
      "blind_metrics":blind,
      "leakage_controls":["future/target/actual columns excluded","7-day temporal purge","train-only robust clipping","blind holdout untouched until final selection","no SMOTE/synthetic oversampling"],
      "limitations":["transaction microdata remain Saudi-localized rather than directly observed Saudi merchant transactions","external real Saudi merchant validation is still required"]
    }
    (OUT/"development_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    summary=(f"# Sales Sentinel V7 — Panel + Blind Holdout\n\n"
             f"- Panel rows: **{len(meta):,}**\n- Features: **{len(features)}**\n- Selected: **{best_name}**\n- Threshold: **{threshold:.2f}**\n"
             f"- Blind rows: **{blind['tp']+blind['fp']+blind['fn']+blind['tn']:,}**; positives: **{blind['tp']+blind['fn']}**\n"
             f"- Blind ROC-AUC: **{100*blind['roc_auc']:.2f}%**\n- Blind PR-AUC: **{100*blind['pr_auc']:.2f}%**\n"
             f"- Precision: **{100*blind['precision']:.2f}%**\n- Recall: **{100*blind['recall']:.2f}%**\n- F1: **{100*blind['f1']:.2f}%**\n"
             f"- GREEN NPV: **{100*blind['green_npv']:.2f}%**\n- Alert rate: **{100*blind['alert_rate']:.2f}%**\n"
             f"- TP / FP / FN / TN: **{blind['tp']} / {blind['fp']} / {blind['fn']} / {blind['tn']}**\n"
             f"- External real Saudi merchant validation: **Pending**\n")
    (OUT/"development_summary.md").write_text(summary,encoding="utf-8")
    pd.DataFrame({"feature":features,"clip_low":lo.values,"clip_high":hi.values}).to_csv(DATA/"feature_bounds_v7.csv",index=False)
    joblib.dump({"version":report["version"],"selected_model":best_name,"threshold":threshold,"features":features,"models":fitted,"clip_low":lo.to_dict(),"clip_high":hi.to_dict()},MOD/"merchant_panel_v7.joblib")
    print(summary)

if __name__=="__main__": main()
