from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports" / "redsea_portable_v16_1"
DEV = BASE / "development_oof.csv"
EXT = BASE / "redsea_predictions.csv"
BASE_REPORT = BASE / "diagnostic_report.json"
OUT = ROOT / "reports" / "redsea_portable_v16_2"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "diagnostic_report.json"
SUMMARY = OUT / "summary.md"
DEV_POLICY = OUT / "development_policy.csv"
EXT_POLICY = OUT / "redsea_policy.csv"
VERSION = "SALES-SENTINEL-V16.2-CAUSAL-PERCENTILE-CALIBRATION"


def metrics(y, pred, score):
    y=np.asarray(y,int); pred=np.asarray(pred,bool); score=np.asarray(score,float)
    tp=int(((y==1)&pred).sum()); fp=int(((y==0)&pred).sum()); fn=int(((y==1)&(~pred)).sum()); tn=int(((y==0)&(~pred)).sum())
    return {
        "accuracy":float(accuracy_score(y,pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y,pred)),
        "precision":float(precision_score(y,pred,zero_division=0)),
        "recall":float(recall_score(y,pred,zero_division=0)),
        "f1":float(f1_score(y,pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y,score)) if len(np.unique(y))==2 else None,
        "pr_auc":float(average_precision_score(y,score)) if len(np.unique(y))==2 else None,
        "alert_rate":float(pred.mean()),
        "green_npv":float(tn/max(tn+fn,1)),
        "tp":tp,"fp":fp,"fn":fn,"tn":tn,
    }


def causal_policy(df:pd.DataFrame, alert_budget:float, lookback:int, warmup:int, fallback_threshold_col:str|None=None, fallback_threshold:float|None=None):
    z=df.sort_values("date").reset_index(drop=True).copy()
    scores=z.score.to_numpy(float)
    pred=np.zeros(len(z),dtype=bool); pct=np.full(len(z),np.nan,float)
    cutoff=1.0-alert_budget
    for i,s in enumerate(scores):
        if i < warmup:
            if fallback_threshold_col is not None:
                t=float(z.loc[i,fallback_threshold_col])
            else:
                t=float(fallback_threshold)
            pred[i]=s>=t
            continue
        a=max(0,i-lookback); ref=scores[a:i]
        # Mid-rank percentile using only scores observed before the current prediction.
        pct[i]=(np.sum(ref<s)+0.5*np.sum(ref==s))/len(ref)
        pred[i]=pct[i]>=cutoff
    z["risk_percentile_past_only"]=pct
    z["policy_prediction"]=pred.astype(int)
    return z


def main():
    dev=pd.read_csv(DEV,parse_dates=["date"])
    ext=pd.read_csv(EXT,parse_dates=["date"])
    base=json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    final_threshold=float(base["development"]["threshold"])

    # Operational alert budget is not tuned on Redsea: inherit the actual V16.1 nested-development alert rate.
    nested_alert=float(base["development"]["nested_oof_metrics"]["alert_rate"])
    candidates=[]
    for lookback in [14,28,56,112,9999]:
        for warmup in [7,14,28]:
            q=causal_policy(dev,nested_alert,lookback,warmup,fallback_threshold_col="threshold")
            m=metrics(q.target,q.policy_prediction,q.score)
            # Development-only objective; favor recall but penalize exceeding inherited alert budget by >5pp.
            penalty=1.5*max(0,m["alert_rate"]-(nested_alert+0.05))
            objective=0.35*m["f1"]+0.25*m["balanced_accuracy"]+0.25*m["recall"]+0.15*m["precision"]-penalty
            candidates.append({"lookback":lookback,"warmup":warmup,"objective":objective,"metrics":m})
    chosen=max(candidates,key=lambda x:(x["objective"],x["metrics"]["f1"],x["metrics"]["recall"]))
    lookback=int(chosen["lookback"]); warmup=int(chosen["warmup"])

    dev_out=causal_policy(dev,nested_alert,lookback,warmup,fallback_threshold_col="threshold")
    dev_m=metrics(dev_out.target,dev_out.policy_prediction,dev_out.score)
    dev_out.to_csv(DEV_POLICY,index=False)

    # External application: the same learned lookback/warmup and inherited alert budget.
    # Warmup uses the already frozen V16.1 final development threshold; afterward only past target-store scores are used.
    ext_out=causal_policy(ext,nested_alert,lookback,warmup,fallback_threshold=final_threshold)
    ext_m=metrics(ext_out.target,ext_out.policy_prediction,ext_out.score)
    ext_out.to_csv(EXT_POLICY,index=False)

    base_ext=base["redsea"]["metrics"]
    report={
        "version":VERSION,
        "status":"POST_OPEN_UNSUPERVISED_CALIBRATION_DIAGNOSTIC",
        "scientific_boundary":"Redsea labels were already open before V16.2, so this is not blind validation. The calibration rule, alert budget, lookback and warmup are determined from development data only. During external inference, percentile calibration uses only risk scores from prior days and never external outcome labels.",
        "policy":{
            "inherited_development_alert_budget":nested_alert,
            "risk_percentile_cutoff":1.0-nested_alert,
            "selected_lookback_days":lookback,
            "selected_warmup_rows":warmup,
            "warmup_threshold":final_threshold,
            "candidate_count":len(candidates),
            "development_candidates":candidates,
        },
        "development":{"metrics":dev_m},
        "redsea":{"metrics":ext_m},
        "comparison_to_v16_1":{
            "external_alert_rate_delta":ext_m["alert_rate"]-base_ext["alert_rate"],
            "external_precision_delta":ext_m["precision"]-base_ext["precision"],
            "external_recall_delta":ext_m["recall"]-base_ext["recall"],
            "external_f1_delta":ext_m["f1"]-base_ext["f1"],
            "external_balanced_accuracy_delta":ext_m["balanced_accuracy"]-base_ext["balanced_accuracy"],
        },
        "red_supported":False,
    }
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    c=report["comparison_to_v16_1"]
    lines=[
        "# Sales Sentinel V16.2 — Causal Percentile Calibration","",
        f"- Status: **{report['status']}**",
        f"- Alert budget inherited from development: **{nested_alert:.2%}**",
        f"- Past-only percentile cutoff: **{1-nested_alert:.2%}**",
        f"- Development-selected lookback / warmup: **{lookback} / {warmup}**","",
        "## Development policy",
        f"- Precision / Recall / F1: **{dev_m['precision']:.2%} / {dev_m['recall']:.2%} / {dev_m['f1']:.2%}**",
        f"- Balanced Accuracy / NPV / Alert: **{dev_m['balanced_accuracy']:.2%} / {dev_m['green_npv']:.2%} / {dev_m['alert_rate']:.2%}**", "",
        "## Redsea post-open diagnostic",
        f"- ROC-AUC / PR-AUC (ranking unchanged): **{ext_m['roc_auc']:.2%} / {ext_m['pr_auc']:.2%}**",
        f"- Precision / Recall / F1: **{ext_m['precision']:.2%} / {ext_m['recall']:.2%} / {ext_m['f1']:.2%}**",
        f"- Accuracy / Balanced Accuracy: **{ext_m['accuracy']:.2%} / {ext_m['balanced_accuracy']:.2%}**",
        f"- NPV / Alert rate: **{ext_m['green_npv']:.2%} / {ext_m['alert_rate']:.2%}**",
        f"- TP/FP/FN/TN: **{ext_m['tp']}/{ext_m['fp']}/{ext_m['fn']}/{ext_m['tn']}**",
        f"- Δ Alert / Precision / Recall / F1 vs V16.1: **{c['external_alert_rate_delta']:+.2%} / {c['external_precision_delta']:+.2%} / {c['external_recall_delta']:+.2%} / {c['external_f1_delta']:+.2%}**","",
        "Scientific note: percentile calibration is causal and label-free at inference; nevertheless Redsea is already post-open and these external numbers remain diagnostic evidence only.",
    ]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(SUMMARY.read_text(encoding="utf-8"))

if __name__=="__main__":
    main()
