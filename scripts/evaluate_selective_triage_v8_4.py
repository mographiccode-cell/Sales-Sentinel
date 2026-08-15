from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
VERSION="SALES-SENTINEL-V8.4-PREQUENTIAL-SELECTIVE-TRIAGE"
SCORES=ROOT/"reports"/"merchant_ensemble_v7_6"/"oof_ensemble_predictions.csv"
OUT=ROOT/"reports"/"merchant_selective_triage_v8_4"; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/"development_report.json"; SUMMARY=OUT/"development_summary.md"; OOF=OUT/"oof_triage.csv"


def choose_thresholds(y,s,min_amber=3,min_green=10):
    y=np.asarray(y,int); s=np.asarray(s,float)
    his=np.linspace(0.20,0.99,160); los=np.linspace(0.01,0.80,160)
    hi=1.01; best_cov=-1
    for t in his:
        ix=s>=t; n=int(ix.sum())
        if n<min_amber: continue
        p=float(y[ix].mean())
        if p>=.80 and n>best_cov: hi=float(t); best_cov=n
    lo=-.01; best_cov=-1
    for t in los:
        ix=s<=t; n=int(ix.sum())
        if n<min_green: continue
        neg=(y[ix]==0); npv=float(neg.mean())
        if npv>=.95 and n>best_cov: lo=float(t); best_cov=n
    if lo>=hi:
        # preserve abstention interval; shrink the less reliable side.
        lo=min(lo,hi-.01)
    return lo,hi


def triage_metrics(y,state,folds):
    y=np.asarray(y,int); state=np.asarray(state); folds=np.asarray(folds,int)
    amber=state=="AMBER"; green=state=="GREEN"; watch=state=="WATCH"; decisive=amber|green
    ap=float(y[amber].mean()) if amber.sum() else None
    gnpv=float((y[green]==0).mean()) if green.sum() else None
    dacc=float(((y[amber]==1).sum()+(y[green]==0).sum())/max(decisive.sum(),1))
    amber_recall=float(((y==1)&amber).sum()/max((y==1).sum(),1))
    rows=[]
    for f in sorted(np.unique(folds)):
        ix=folds==f; yy=y[ix]; ss=state[ix]; aa=ss=="AMBER"; gg=ss=="GREEN"; dd=aa|gg
        rows.append({"fold_id":int(f),"rows":int(ix.sum()),"amber_n":int(aa.sum()),"green_n":int(gg.sum()),"watch_n":int((ss=="WATCH").sum()),"amber_precision":float(yy[aa].mean()) if aa.sum() else None,"green_npv":float((yy[gg]==0).mean()) if gg.sum() else None,"decisive_accuracy":float(((yy[aa]==1).sum()+(yy[gg]==0).sum())/max(dd.sum(),1)),"decisive_coverage":float(dd.mean())})
    return {"amber_precision":ap,"green_npv":gnpv,"amber_recall":amber_recall,"amber_n":int(amber.sum()),"green_n":int(green.sum()),"watch_n":int(watch.sum()),"decisive_n":int(decisive.sum()),"decisive_coverage":float(decisive.mean()),"watch_rate":float(watch.mean()),"decisive_accuracy":dacc,"per_fold":rows}


def main():
    d=pd.read_csv(SCORES).sort_values(["fold_id","date"]).reset_index(drop=True)
    y=d.y.to_numpy(int); s=d.ensemble_score.to_numpy(float); folds=d.fold_id.to_numpy(int)
    state=np.full(len(d),"WATCH",dtype=object); details=[]
    for f in sorted(np.unique(folds)):
        cur=folds==f
        if f==0:
            details.append({"fold_id":0,"mode":"calibration_burn_in","low":None,"high":None}); continue
        hist=folds<f
        lo,hi=choose_thresholds(y[hist],s[hist])
        st=np.full(cur.sum(),"WATCH",dtype=object); st[s[cur]<=lo]="GREEN"; st[s[cur]>=hi]="AMBER"; state[cur]=st
        hm=triage_metrics(y[hist],np.where(s[hist]<=lo,"GREEN",np.where(s[hist]>=hi,"AMBER","WATCH")),folds[hist])
        details.append({"fold_id":int(f),"history_rows":int(hist.sum()),"low":lo,"high":hi,"history_amber_precision":hm["amber_precision"],"history_green_npv":hm["green_npv"],"history_decisive_coverage":hm["decisive_coverage"]})
    m=triage_metrics(y,state,folds)
    post=folds>0; postm=triage_metrics(y[post],state[post],folds[post])

    # Development oracle is diagnostic only.
    best=None
    for lo in np.linspace(.05,.65,121):
        for hi in np.linspace(max(lo+.05,.35),.98,100):
            st=np.where(s<=lo,"GREEN",np.where(s>=hi,"AMBER","WATCH")); z=triage_metrics(y,st,folds)
            ok=(z["amber_precision"] is not None and z["amber_precision"]>=.80 and z["green_npv"] is not None and z["green_npv"]>=.95)
            key=(int(ok),z["decisive_coverage"],z["decisive_accuracy"],z["amber_recall"])
            if best is None or key>best[0]: best=(key,float(lo),float(hi),z,ok)
    oracle={"low":best[1],"high":best[2],"metrics":best[3],"meets_80_precision_95_npv":bool(best[4])}
    report={"version":VERSION,"status":"DEVELOPMENT_SELECTIVE_POLICY","scientific_boundary":"Fold 0 is calibration burn-in and emits WATCH only. Folds 1-4 choose GREEN/AMBER thresholds exclusively from earlier folds. Selective metrics must not be confused with full-coverage binary precision/recall; WATCH is an explicit abstention state. External Saudi merchant validation remains required.","all_folds":m,"post_calibration_folds_1_4":postm,"fold_details":details,"development_oracle":oracle,"targets":{"amber_precision":.80,"green_npv":.95},"red_supported":False}
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    q=d[["date","y","fold_id","ensemble_score"]].copy(); q["triage_state"]=state; q.to_csv(OOF,index=False)
    def fmt(v): return "N/A" if v is None else f"{v:.2%}"
    lines=["# Sales Sentinel V8.4 — Prequential Selective Triage","", "- Fold 0: **WATCH-only calibration burn-in**", "- Folds 1-4 thresholds: **learned from earlier folds only**","", "## Post-calibration folds 1-4",f"- AMBER precision: **{fmt(postm['amber_precision'])}**",f"- GREEN NPV: **{fmt(postm['green_npv'])}**",f"- Decisive accuracy: **{postm['decisive_accuracy']:.2%}**",f"- Decisive coverage: **{postm['decisive_coverage']:.2%}**",f"- WATCH rate: **{postm['watch_rate']:.2%}**",f"- AMBER recall of all declines: **{postm['amber_recall']:.2%}**",f"- AMBER/GREEN/WATCH counts: **{postm['amber_n']}/{postm['green_n']}/{postm['watch_n']}**","",f"- Development oracle can meet AMBER>=80% and GREEN NPV>=95%: **{oracle['meets_80_precision_95_npv']}**",f"- Oracle decisive coverage: **{oracle['metrics']['decisive_coverage']:.2%}**","","Important: selective triage does not make full-coverage binary Precision and Recall both exceed 80%; uncertain cases are explicitly routed to WATCH."]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
