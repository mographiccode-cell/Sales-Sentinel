from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import train_merchant_error_corrector_v7_5 as v75

ROOT=Path(__file__).resolve().parents[1]
VERSION="SALES-SENTINEL-V8.1-PREQUENTIAL-REBOUND-FILTER"
V75=ROOT/"reports"/"merchant_error_corrector_v7_5"/"oof_fused_predictions.csv"
V8=ROOT/"reports"/"merchant_target_refinement_v8"/"oof_fused_predictions.csv"
DIAG=ROOT/"reports"/"merchant_market_fusion_v6_1"/"oof_policy_diagnostics.csv"
OUT=ROOT/"reports"/"merchant_rebound_filter_v8_1"; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/"development_report.json"; SUMMARY=OUT/"development_summary.md"; OOF=OUT/"oof_predictions.csv"


def rules():
    scopes=["any","market","quiet","nonstrong"]
    qvs=np.round(np.arange(.05,.61,.05),2)
    rescopes=["none","market","near","any"]
    qrs=[1.01,.75,.80,.85,.90,.93,.96]
    out=[("none",0.0,"none",1.01)]
    for s,q,rs,qr in product(scopes,qvs,rescopes,qrs):
        if rs=="none" and qr!=1.01: continue
        if rs!="none" and qr>1: continue
        out.append((s,float(q),rs,float(qr)))
    return out


def apply(base,strong,quiet,market,score,mm,mp,rule):
    s,q,rs,qr=rule; p=base.copy()
    if s!="none":
        if s=="market": mask=market&(~strong)
        elif s=="quiet": mask=quiet&(~strong)&(~market)
        elif s=="nonstrong": mask=~strong
        else: mask=np.ones(len(p),bool)
        p=p & ~(base&mask&(score<q))
    if rs!="none":
        if rs=="market": mask=mp>=.10
        elif rs=="near": mask=mm>=.28
        else: mask=np.ones(len(p),bool)
        p=p | ((~base)&mask&(score>=qr))
    return p


def prequential(y,folds,base,strong,quiet,market,score,mm,mp):
    final=base.copy(); details=[]; grid=rules()
    for f in sorted(np.unique(folds)):
        cur=folds==f
        if f==0:
            details.append({"fold_id":int(f),"mode":"v7_5_bootstrap"}); continue
        hist=folds<f; bm=v75.metrics(y[hist],base[hist],folds[hist]); best=None
        for r in grid:
            pp=apply(base[hist],strong[hist],quiet[hist],market[hist],score[hist],mm[hist],mp[hist],r)
            m=v75.metrics(y[hist],pp,folds[hist])
            feasible=(m["recall"]>=bm["recall"]-.005 and m["green_npv"]>=bm["green_npv"]-.002 and m["fp"]<=bm["fp"] and m["f1"]>=bm["f1"] and m["alert_rate"]<=bm["alert_rate"])
            key=(int(feasible),m["f1"],m["precision"],-m["fp"],m["recall"],m["green_npv"])
            if best is None or key>best[0]: best=(key,r,m,feasible)
        _,r,hm,ok=best
        if not ok: r=("none",0.0,"none",1.01)
        final[cur]=apply(base[cur],strong[cur],quiet[cur],market[cur],score[cur],mm[cur],mp[cur],r)
        details.append({"fold_id":int(f),"rule":list(r),"history_feasible":bool(ok),"history_metrics":hm})
    return final,details


def main():
    a=pd.read_csv(V75).sort_values(["fold_id","date"]).reset_index(drop=True)
    b=pd.read_csv(V8).sort_values(["fold_id","date"]).reset_index(drop=True)
    d=pd.read_csv(DIAG)
    if len(a)!=381 or len(b)!=381 or not np.array_equal(a.y.to_numpy(int),b.y.to_numpy(int)): raise RuntimeError("OOF mismatch")
    y=a.y.to_numpy(int); folds=a.fold_id.to_numpy(int); base=a.v7_5_pred.to_numpy(bool); score=b.v8_rank.to_numpy(float)
    strong,quiet,market,_=v75.base_components(d); mm=d.merchant_mean.to_numpy(float); mp=d.market_v3__risk_p90.to_numpy(float)
    bm=v75.metrics(y,base,folds)
    pred,details=prequential(y,folds,base,strong,quiet,market,score,mm,mp); m=v75.metrics(y,pred,folds)

    # Development oracle only diagnoses whether the score contains usable rebound information.
    best=None
    for r in rules():
        p=apply(base,strong,quiet,market,score,mm,mp,r); z=v75.metrics(y,p,folds)
        feasible=(z["recall"]>=bm["recall"] and z["green_npv"]>=bm["green_npv"] and z["fp"]<bm["fp"] and z["f1"]>bm["f1"] and z["worst_fold_recall"]>=bm["worst_fold_recall"])
        key=(int(feasible),z["f1"],z["precision"],-z["fp"],z["recall"])
        if best is None or key>best[0]: best=(key,r,z,feasible)
    oracle={"rule":list(best[1]),"metrics":best[2],"strictly_dominates_v7_5":bool(best[3])}

    adopt=bool(m["recall"]>=bm["recall"] and m["green_npv"]>=bm["green_npv"] and m["f1"]>bm["f1"] and m["precision"]>bm["precision"] and m["fp"]<bm["fp"] and m["worst_fold_recall"]>=bm["worst_fold_recall"])
    report={"version":VERSION,"status":"DEVELOPMENT_BEST" if adopt else "EXPERIMENTAL_V7_5_REMAINS_BEST","scientific_boundary":"V8.1 uses V7.5 as the frozen base decision and learns rebound-filter rules only from earlier OOF folds for each subsequent fold. The oracle is diagnostic only. Fresh Saudi merchant validation remains required.","v7_5":bm,"v8_1":m,"prequential_details":details,"development_oracle":oracle,"adopt_over_v7_5":adopt,"red_supported":False}
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    pd.DataFrame({"date":a.date,"y":y,"fold_id":folds,"v7_5_pred":base.astype(int),"v8_rank":score,"v8_1_pred":pred.astype(int)}).to_csv(OOF,index=False)
    lines=["# Sales Sentinel V8.1 — Prequential Rebound Filter","",f"- Status: **{report['status']}**",f"- Precision: V7.5 **{bm['precision']:.2%}** -> V8.1 **{m['precision']:.2%}**",f"- Recall: V7.5 **{bm['recall']:.2%}** -> V8.1 **{m['recall']:.2%}**",f"- F1: V7.5 **{bm['f1']:.2%}** -> V8.1 **{m['f1']:.2%}**",f"- NPV: V7.5 **{bm['green_npv']:.2%}** -> V8.1 **{m['green_npv']:.2%}**",f"- Alert rate: V7.5 **{bm['alert_rate']:.2%}** -> V8.1 **{m['alert_rate']:.2%}**",f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",f"- Adopt over V7.5: **{adopt}**","",f"- Development oracle strictly dominates V7.5: **{oracle['strictly_dominates_v7_5']}**",f"- Oracle rule: **{oracle['rule']}**",f"- Oracle TP/FP/FN/TN: **{oracle['metrics']['tp']}/{oracle['metrics']['fp']}/{oracle['metrics']['fn']}/{oracle['metrics']['tn']}**","","Scientific boundary: prequential development evidence only; oracle metrics are not deployment evidence."]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
