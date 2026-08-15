from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import train_merchant_category_signals_v7_1 as v71
import evaluate_redsea_portable_v16 as v16
import evaluate_redsea_portable_v16_1 as v161

ROOT=Path(__file__).resolve().parents[1]
REDSEA_FILE=Path(os.environ.get("REDSEA_FILE","/tmp/redsea_mendeley/RedSea_Data_Cleaned.xlsx"))
BASE_REPORT=ROOT/"reports"/"redsea_portable_v16_1"/"diagnostic_report.json"
BASE_PRED=ROOT/"reports"/"redsea_portable_v16_1"/"redsea_predictions.csv"
V162_POLICY=ROOT/"reports"/"redsea_portable_v16_2"/"redsea_policy.csv"
OUT=ROOT/"reports"/"redsea_online_adaptation_v17"
OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/"adaptation_report.json"; SUMMARY=OUT/"summary.md"; PREDS=OUT/"prequential_predictions.csv"
VERSION="SALES-SENTINEL-V17-CAUSAL-WEEKLY-MERCHANT-ADAPTATION"
LABEL_DELAY_DAYS=7
MIN_TARGET_LABELS=14
TARGET_SAMPLE_WEIGHT=4.0
RETRAIN_EVERY_NEW_LABELS=7


def bin_metrics(y,pred,score):
    y=np.asarray(y,int); p=np.asarray(pred,bool); s=np.asarray(score,float)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&(~p)).sum()); tn=int(((y==0)&(~p)).sum())
    return {
        "accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),
        "precision":float(precision_score(y,p,zero_division=0)),"recall":float(recall_score(y,p,zero_division=0)),"f1":float(f1_score(y,p,zero_division=0)),
        "roc_auc":float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None,"pr_auc":float(average_precision_score(y,s)) if len(np.unique(y))==2 else None,
        "alert_rate":float(p.mean()),"green_npv":float(tn/max(tn+fn,1)),"tp":tp,"fp":fp,"fn":fn,"tn":tn,
    }


def causal_rank_alert(scores:list[float], alert_budget:float, lookback:int=112, warmup:int=7, fallback_threshold:float=.242):
    i=len(scores)-1; s=scores[-1]
    if i<warmup: return bool(s>=fallback_threshold), np.nan
    ref=np.asarray(scores[max(0,i-lookback):i],float)
    pct=float((np.sum(ref<s)+.5*np.sum(ref==s))/len(ref))
    return bool(pct>=1-alert_budget),pct


def main():
    base=json.loads(BASE_REPORT.read_text(encoding="utf-8"))
    alert_budget=float(base["development"]["nested_oof_metrics"]["alert_rate"])
    fallback_threshold=float(base["development"]["threshold"])

    dev_daily=v16.source_daily(); ext_daily=v16.redsea_daily()
    dev_meta,Xdev0,_=v16.build_meta_and_features(dev_daily)
    ext_meta,Xext0,_=v16.build_meta_and_features(ext_daily)
    Xdev,cols=v161.filter_comparable(Xdev0); Xext,cols2=v161.filter_comparable(Xext0)
    if cols!=cols2 or len(ext_meta)!=len(Xext): raise RuntimeError("V17 schema/alignment mismatch")

    # Fixed model family inherited from V16.1. No model/weight/hyperparameter search on Redsea.
    factory=v71.factories()["extra_trees"]
    ydev=dev_meta.target.astype(int).reset_index(drop=True)
    base_pred=pd.read_csv(BASE_PRED,parse_dates=["date"])
    p162=pd.read_csv(V162_POLICY,parse_dates=["date"])
    lookup_base=base_pred.set_index("date")
    lookup_162=p162.set_index("date")

    rows=[]; model=None; prep=None; last_fit_n=-10**9; adapted_scores=[]
    for i,row in ext_meta.iterrows():
        date=pd.Timestamp(row.date)
        # A label for target date d is assumed available only after d+7 days.
        available=(ext_meta.date <= date-pd.Timedelta(days=LABEL_DELAY_DAYS))
        target_idx=np.where(available.to_numpy(bool))[0]
        n_avail=len(target_idx)
        active=n_avail>=MIN_TARGET_LABELS
        retrained=False
        if active and (model is None or n_avail-last_fit_n>=RETRAIN_EVERY_NEW_LABELS):
            Xtr_raw=pd.concat([Xdev, Xext.iloc[target_idx]],ignore_index=True)
            ytr=pd.concat([ydev, ext_meta.iloc[target_idx].target.astype(int).reset_index(drop=True)],ignore_index=True)
            Xtr,_,prep=v71.fold_prepare(Xtr_raw,Xtr_raw)
            sw=np.ones(len(ytr),float); sw[len(ydev):]=TARGET_SAMPLE_WEIGHT
            model=clone(factory)
            model.fit(Xtr,ytr,sample_weight=sw)
            last_fit_n=n_avail; retrained=True
        if active:
            # Apply the preprocessing fitted using source + past labeled target rows only.
            q=Xext.iloc[[i]].copy()
            for c in q.columns:
                m=prep[c]
                q[c]=pd.to_numeric(q[c],errors="coerce").clip(m["p01"],m["p99"]).fillna(m["median"])
            score=float(model.predict_proba(q.astype(float))[:,1][0])
            adapted_scores.append(score)
            policy,pct=causal_rank_alert(adapted_scores,alert_budget,lookback=112,warmup=7,fallback_threshold=fallback_threshold)
        else:
            score=np.nan; policy=False; pct=np.nan
        b=lookup_base.loc[date]; c=lookup_162.loc[date]
        rows.append({
            "date":date,"target":int(row.target),"future_ratio":float(row.future_ratio),
            "available_target_labels":n_avail,"adaptation_active":active,"retrained_today":retrained,
            "adapted_score":score,"adapted_risk_percentile":pct,"adapted_alert":int(policy),
            "v16_1_static_score":float(b.score),"v16_1_static_alert":int(b.prediction),
            "v16_2_alert":int(c.policy_prediction),
        })
    out=pd.DataFrame(rows); out.to_csv(PREDS,index=False)
    ev=out[out.adaptation_active].copy()
    if len(ev)<20: raise RuntimeError(f"Too few adapted evaluation rows: {len(ev)}")
    adapted=bin_metrics(ev.target,ev.adapted_alert,ev.adapted_score)
    static=bin_metrics(ev.target,ev.v16_1_static_alert,ev.v16_1_static_score)
    # V16.2 score/ranking is the same V16.1 score, decision is percentile policy.
    cal=bin_metrics(ev.target,ev.v16_2_alert,ev.v16_1_static_score)

    report={
        "version":VERSION,"status":"POST_OPEN_CAUSAL_ADAPTATION_DIAGNOSTIC",
        "scientific_boundary":"Redsea was already opened in earlier diagnostics; V17 is not independent validation. It tests a deployable prequential adaptation protocol. Every prediction is fit only with localized source data plus Redsea labels whose complete 7-day future was already observable before the prediction date. No current/future Redsea label enters its own or an earlier prediction.",
        "protocol":{"label_delay_days":LABEL_DELAY_DAYS,"minimum_past_target_labels":MIN_TARGET_LABELS,"target_sample_weight":TARGET_SAMPLE_WEIGHT,"retrain_every_new_labels":RETRAIN_EVERY_NEW_LABELS,"model":"extra_trees inherited from V16.1","alert_budget_inherited_from_development":alert_budget,"percentile_lookback":112,"percentile_warmup":7},
        "evaluation":{"rows":len(ev),"date_start":str(ev.date.min().date()),"date_end":str(ev.date.max().date()),"positive_rate":float(ev.target.mean()),"adapted":adapted,"v16_1_static_same_rows":static,"v16_2_calibrated_same_rows":cal},
        "deltas_vs_static":{"auc":adapted["roc_auc"]-static["roc_auc"],"pr_auc":adapted["pr_auc"]-static["pr_auc"],"precision":adapted["precision"]-static["precision"],"recall":adapted["recall"]-static["recall"],"f1":adapted["f1"]-static["f1"],"alert_rate":adapted["alert_rate"]-static["alert_rate"]},
        "red_supported":False,
    }
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    a=adapted;s=static;c=cal;d=report["deltas_vs_static"]
    lines=["# Sales Sentinel V17 — Causal Weekly Merchant Adaptation","",f"- Status: **{report['status']}**",f"- Evaluation after enough past labels: **{len(ev)} rows ({report['evaluation']['date_start']} → {report['evaluation']['date_end']})**",f"- Evaluation decline prevalence: **{report['evaluation']['positive_rate']:.2%}**",f"- Target labels available with: **{LABEL_DELAY_DAYS}-day delay**",f"- Adaptation target weight: **{TARGET_SAMPLE_WEIGHT:.1f}x**", "","## V17 adapted",f"- ROC-AUC / PR-AUC: **{a['roc_auc']:.2%} / {a['pr_auc']:.2%}**",f"- Precision / Recall / F1: **{a['precision']:.2%} / {a['recall']:.2%} / {a['f1']:.2%}**",f"- Balanced Accuracy / NPV / Alert: **{a['balanced_accuracy']:.2%} / {a['green_npv']:.2%} / {a['alert_rate']:.2%}**",f"- TP/FP/FN/TN: **{a['tp']}/{a['fp']}/{a['fn']}/{a['tn']}**","","## Same-date controls",f"- V16.1 static AUC / P / R / F1 / Alert: **{s['roc_auc']:.2%} / {s['precision']:.2%} / {s['recall']:.2%} / {s['f1']:.2%} / {s['alert_rate']:.2%}**",f"- V16.2 calibrated P / R / F1 / Alert: **{c['precision']:.2%} / {c['recall']:.2%} / {c['f1']:.2%} / {c['alert_rate']:.2%}**",f"- Δ V17 vs static AUC / Recall / F1 / Alert: **{d['auc']:+.2%} / {d['recall']:+.2%} / {d['f1']:+.2%} / {d['alert_rate']:+.2%}**","","Scientific note: this is a post-open prequential adaptation experiment, not fresh external validation. Its value is testing whether a real deployed merchant can improve after accumulating its own labeled history without future leakage."]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(SUMMARY.read_text(encoding="utf-8"))

if __name__=="__main__": main()
