from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
VERSION="SALES-SENTINEL-V7.6-RANK-ENSEMBLE-ERROR-CORRECTOR"
V75_REPORT=ROOT/"reports"/"merchant_error_corrector_v7_5"/"development_report.json"
V75_OOF=ROOT/"reports"/"merchant_error_corrector_v7_5"/"oof_candidate_scores.csv"
V71_OOF=ROOT/"reports"/"merchant_category_signals_v7_1"/"oof_predictions.csv"
V61_DIAG=ROOT/"reports"/"merchant_market_fusion_v6_1"/"oof_policy_diagnostics.csv"
V61_REPORT=ROOT/"reports"/"merchant_market_fusion_v6_1"/"development_report.json"
OUT=ROOT/"reports"/"merchant_ensemble_v7_6"; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/"development_report.json"; SUMMARY=OUT/"development_summary.md"; OOF=OUT/"oof_ensemble_predictions.csv"


def pct_by_fold(x,folds):
    x=np.asarray(x,float); folds=np.asarray(folds,int); out=np.zeros(len(x),float)
    for f in np.unique(folds):
        ix=np.where(folds==f)[0]; vals=x[ix]; order=np.argsort(np.argsort(vals,kind="mergesort"),kind="mergesort"); out[ix]=(order+1)/len(ix)
    return out


def base_components(d):
    mm=d.merchant_mean.to_numpy(float); lr=d.merchant_logreg.to_numpy(float); ex=d.merchant_extra.to_numpy(float); dis=d.merchant_disagreement.to_numpy(float); mp=d.market_v3__risk_p90.to_numpy(float)
    strong=(mm>=.70)&(ex>=.60); quiet=(mp<=.05)&(lr>=.45)&(dis>=.10)&(mm>=.35); market=(mp>=.20)&(mm>=.35)
    return strong,quiet,market,strong|quiet|market


def metrics(y,p,folds):
    y=np.asarray(y,int); p=np.asarray(p,bool); folds=np.asarray(folds,int)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&(~p)).sum()); tn=int(((y==0)&(~p)).sum())
    per=[]
    for f in np.unique(folds):
        ix=folds==f; yy=y[ix]; pp=p[ix]; a=int(((yy==1)&pp).sum()); b=int(((yy==0)&pp).sum()); c=int(((yy==1)&(~pp)).sum()); dd=int(((yy==0)&(~pp)).sum())
        per.append({"fold_id":int(f),"recall":a/max(a+c,1),"precision":a/max(a+b,1),"f1":2*a/max(2*a+b+c,1),"alert_rate":float(pp.mean()),"green_npv":dd/max(dd+c,1)})
    return {"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"f1":2*tp/max(2*tp+fp+fn,1),"accuracy":(tp+tn)/len(y),"balanced_accuracy":.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)),"green_npv":tn/max(tn+fn,1),"alert_rate":float(p.mean()),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"worst_fold_recall":min(x["recall"] for x in per),"max_fold_alert_rate":max(x["alert_rate"] for x in per),"per_fold":per}


def rules():
    out=[]
    for scope,qv,rs,qr in product(["none","quiet","market","nonstrong","any"],[0,.1,.2,.3,.4,.5],["none","near","market","any"],[1.01,.80,.85,.90,.93,.96]):
        if scope=="none" and qv!=0: continue
        if scope!="none" and qv==0: continue
        if rs=="none" and qr!=1.01: continue
        if rs!="none" and qr>1: continue
        out.append((scope,qv,rs,qr))
    return out


def apply(base,strong,quiet,market,score,mm,mp,r):
    scope,qv,rs,qr=r; p=base.copy()
    if scope!="none":
        if scope=="quiet": mask=quiet&(~strong)&(~market)
        elif scope=="market": mask=market&(~strong)
        elif scope=="nonstrong": mask=~strong
        else: mask=np.ones(len(p),bool)
        p=p&~(base&mask&(score<qv))
    if rs!="none":
        if rs=="near": mask=mm>=.28
        elif rs=="market": mask=mp>=.10
        else: mask=np.ones(len(p),bool)
        p=p|((~base)&mask&(score>=qr))
    return p


def prequential(y,folds,base,strong,quiet,market,score,mm,mp):
    final=base.copy(); detail=[]; grid=rules()
    for f in np.unique(folds):
        cur=folds==f
        if f==0: detail.append({"fold_id":0,"mode":"v6_1_bootstrap"}); continue
        hist=folds<f; bm=metrics(y[hist],base[hist],folds[hist]); best=None
        for r in grid:
            p=apply(base[hist],strong[hist],quiet[hist],market[hist],score[hist],mm[hist],mp[hist],r); m=metrics(y[hist],p,folds[hist])
            feas=m["recall"]>=bm["recall"]-.02 and m["green_npv"]>=bm["green_npv"]-.005 and m["alert_rate"]<=bm["alert_rate"]+.01 and m["fp"]<=bm["fp"] and m["f1"]>=bm["f1"]
            key=(int(feas),m["f1"],m["precision"],-m["fp"],m["recall"])
            if best is None or key>best[0]: best=(key,r,m,feas)
        r=best[1] if best[3] else ("none",0,"none",1.01)
        final[cur]=apply(base[cur],strong[cur],quiet[cur],market[cur],score[cur],mm[cur],mp[cur],r)
        detail.append({"fold_id":int(f),"history_rule":list(r),"history_feasible":bool(best[3])})
    return final,detail


def frontier(y,scores):
    best_p={"precision":0.0,"recall":0.0,"threshold":None}; best_r={"precision":0.0,"recall":0.0,"threshold":None}; best_f1={"f1":0.0}
    both=False
    for t in np.unique(np.r_[np.linspace(0,1,501),scores]):
        p=scores>=t; tp=((y==1)&p).sum(); fp=((y==0)&p).sum(); fn=((y==1)&(~p)).sum()
        precision=float(tp/max(tp+fp,1)); recall=float(tp/max(tp+fn,1)); f1=float(2*tp/max(2*tp+fp+fn,1))
        if recall>=.80 and precision>best_p["precision"]: best_p={"precision":precision,"recall":recall,"threshold":float(t)}
        if precision>=.80 and recall>best_r["recall"]: best_r={"precision":precision,"recall":recall,"threshold":float(t)}
        if f1>best_f1["f1"]: best_f1={"f1":f1,"precision":precision,"recall":recall,"threshold":float(t)}
        if precision>=.80 and recall>=.80: both=True
    return {"max_precision_at_recall_ge_80":best_p,"max_recall_at_precision_ge_80":best_r,"best_f1_threshold":best_f1,"precision_and_recall_both_ge_80_exists":both}


def main():
    v75=json.loads(V75_REPORT.read_text(encoding="utf-8")); selected_id=int(v75["selected"]["config_id"]); v75m=v75["selected"]["prequential_fusion"]
    a=pd.read_csv(V75_OOF); a=a[a.config_id==selected_id].sort_values(["fold_id","date"]).reset_index(drop=True)
    b=pd.read_csv(V71_OOF); b=b[(b.scope=="merchant_plus_category_signals")&(b.model=="mean_ensemble")].sort_values(["fold_id","date"]).reset_index(drop=True)
    d=pd.read_csv(V61_DIAG); y=d.y.to_numpy(int); folds=d.fold_id.to_numpy(int); mm=d.merchant_mean.to_numpy(float); mp=d.market_v3__risk_p90.to_numpy(float)
    if len(a)!=381 or len(b)!=381 or not np.array_equal(a.target.to_numpy(int),y) or not np.array_equal(b.target.to_numpy(int),y): raise RuntimeError("OOF alignment mismatch")
    strong,quiet,market,base=base_components(d); base_m=metrics(y,base,folds)
    r75=pct_by_fold(a.rank_score.to_numpy(float),folds); r71=pct_by_fold(b.score.to_numpy(float),folds); r6=pct_by_fold(mm,folds)

    candidates=[]
    for w75 in np.arange(0,1.001,.1):
        for w71 in np.arange(0,1.001-w75,.1):
            w6=1-w75-w71
            if w6<-.001: continue
            score=w75*r75+w71*r71+w6*r6
            auc=float(roc_auc_score(y,score)); pr=float(average_precision_score(y,score)); min_auc=min(float(roc_auc_score(y[folds==f],score[folds==f])) for f in np.unique(folds))
            pred,detail=prequential(y,folds,base,strong,quiet,market,score,mm,mp); m=metrics(y,pred,folds)
            strict=(auc>=v75["selected"]["ranking"]["roc_auc"] and pr>=float(json.loads(V61_REPORT.read_text(encoding="utf-8"))["ranking"]["pr_auc"]) and m["precision"]>v75m["precision"] and m["recall"]>=v75m["recall"] and m["f1"]>v75m["f1"] and m["green_npv"]>=v75m["green_npv"] and m["alert_rate"]<=v75m["alert_rate"] and m["worst_fold_recall"]>=.60 and m["fp"]<v75m["fp"])
            candidates.append({"weights":{"v7_5":float(w75),"v7_1":float(w71),"v6_merchant":float(w6)},"roc_auc":auc,"pr_auc":pr,"min_fold_auc":min_auc,"metrics":m,"details":detail,"strict_improvement_over_v7_5":bool(strict),"score":score})
    def key(c):
        m=c["metrics"]; return (int(c["strict_improvement_over_v7_5"]),m["f1"],m["precision"],m["recall"],c["pr_auc"],c["roc_auc"],-m["fp"])
    sel=max(candidates,key=key); score=sel.pop("score"); adopt=bool(sel["strict_improvement_over_v7_5"])
    # Remove arrays from other candidates for JSON.
    clean=[]
    for c in candidates:
        c.pop("score",None); clean.append(c)
    feas=frontier(y,score)
    pd.DataFrame({"date":a.date,"y":y,"fold_id":folds,"ensemble_score":score}).to_csv(OOF,index=False)
    report={"version":VERSION,"status":"DEVELOPMENT_ADOPTABLE_PENDING_EXTERNAL_VALIDATION" if adopt else "EXPERIMENTAL_V7_5_REMAINS_BEST","scientific_boundary":"V7.6 is a development-only rank ensemble over previously generated OOF evidence. It does not create independent validation. The feasibility frontier is diagnostic and must not be presented as external performance.","v6_1":base_m,"v7_5":v75m,"selected":sel,"candidate_count":len(clean),"feasibility_frontier":feas,"adopt_over_v7_5":adopt,"top_candidates":sorted(clean,key=key,reverse=True)[:10]}
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    m=sel["metrics"]
    lines=["# Sales Sentinel V7.6 — Rank Ensemble Error Corrector","",f"- Status: **{report['status']}**",f"- Weights: **{sel['weights']}**",f"- ROC-AUC / PR-AUC: **{sel['roc_auc']:.2%} / {sel['pr_auc']:.2%}**",f"- Minimum fold AUC: **{sel['min_fold_auc']:.2%}**","",f"- V7.5 Precision / Recall / F1: **{v75m['precision']:.2%} / {v75m['recall']:.2%} / {v75m['f1']:.2%}**",f"- V7.6 Precision / Recall / F1: **{m['precision']:.2%} / {m['recall']:.2%} / {m['f1']:.2%}**",f"- V7.6 NPV / Alert rate: **{m['green_npv']:.2%} / {m['alert_rate']:.2%}**",f"- V7.6 TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",f"- Adopt over V7.5: **{adopt}**","",f"- Max precision while recall >=80%: **{feas['max_precision_at_recall_ge_80']['precision']:.2%}**",f"- Max recall while precision >=80%: **{feas['max_recall_at_precision_ge_80']['recall']:.2%}**",f"- Precision and recall both >=80% exists: **{feas['precision_and_recall_both_ge_80_exists']}**","","Scientific boundary: development-only evidence; fresh Saudi merchant validation remains required."]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
