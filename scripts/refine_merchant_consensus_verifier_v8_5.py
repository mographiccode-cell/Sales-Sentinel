from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import train_merchant_error_corrector_v7_5 as v75
import train_merchant_alert_verifier_v8_2 as v82

ROOT=Path(__file__).resolve().parents[1]
VERSION="SALES-SENTINEL-V8.5-PREQUENTIAL-CONSENSUS-VERIFIER"
V76=ROOT/"reports"/"merchant_ensemble_v7_6"/"oof_ensemble_predictions.csv"
OUT=ROOT/"reports"/"merchant_consensus_verifier_v8_5"; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/"development_report.json"; SUMMARY=OUT/"development_summary.md"; OOF=OUT/"oof_predictions.csv"


def build():
    a,b,d,x,strong,quiet,market=v82.build_oof_features()
    e=pd.read_csv(V76).sort_values(["fold_id","date"]).reset_index(drop=True)
    if len(e)!=len(a) or not np.array_equal(e.y.to_numpy(int),a.y.to_numpy(int)): raise RuntimeError("V7.6 alignment mismatch")
    x=x.copy(); x["v76_ensemble"]=e.ensemble_score.astype(float)
    x["rank_consensus_mean"]=(x.v75_rank+x.v8_rank+x.v76_ensemble)/3
    x["rank_consensus_min"]=x[["v75_rank","v8_rank","v76_ensemble"]].min(axis=1)
    x["rank_consensus_max"]=x[["v75_rank","v8_rank","v76_ensemble"]].max(axis=1)
    x["rank_disagreement"]=x.rank_consensus_max-x.rank_consensus_min
    x["v76_minus_v8"]=x.v76_ensemble-x.v8_rank
    return a,b,e,d,x,strong,quiet,market


def fit_predict(Xtr,ytr,Xva,C):
    sc=StandardScaler(); A=sc.fit_transform(Xtr); B=sc.transform(Xva)
    m=LogisticRegression(C=C,class_weight="balanced",max_iter=2500,solver="liblinear",random_state=42)
    m.fit(A,ytr); return m.predict_proba(A)[:,1],m.predict_proba(B)[:,1]


def candidate(a,b,e,x,strong,quiet,market,cfg):
    y=a.y.to_numpy(int); folds=a.fold_id.to_numpy(int); base=a.v7_5_pred.to_numpy(bool)
    v8r=b.v8_rank.to_numpy(float); v76=e.ensemble_score.to_numpy(float); guard_score=np.minimum(v8r,v76)
    final=base.copy(); details=[]
    cols=[c for c in x.columns if not c.startswith("v8__")] if cfg["features"]=="core" else list(x.columns)
    for f in sorted(np.unique(folds)):
        cur=folds==f
        if f==0: details.append({"fold_id":0,"mode":"bootstrap"}); continue
        hist=folds<f; ha=hist&base
        if ha.sum()<20 or len(np.unique(y[ha]))<2: continue
        ptr,pcur=fit_predict(x.loc[ha,cols],y[ha],x.loc[cur,cols],cfg["C"]); tp=ptr[y[ha]==1]
        thr=float(max(0,np.quantile(tp,cfg["tp_quantile"])-cfg["margin"]))
        if cfg["scope"]=="market": scope=market[cur]&(~strong[cur])
        elif cfg["scope"]=="nonstrong": scope=~strong[cur]
        else: scope=np.ones(cur.sum(),bool)
        cb=base[cur].copy(); veto=cb&scope&(pcur<thr)&(guard_score[cur]<cfg["rank_guard"]); cp=cb&(~veto)
        if cfg["rescue"]<=1:
            cp=cp|((~cb)&market[cur]&(v76[cur]>=cfg["rescue"])&(v8r[cur]>=cfg["rescue"]-.10))
        final[cur]=cp
        details.append({"fold_id":int(f),"threshold":thr,"history_alerts":int(ha.sum()),"history_tp":int(y[ha].sum()),"vetoes":int(veto.sum())})
    return final,details


def main():
    a,b,e,d,x,strong,quiet,market=build(); y=a.y.to_numpy(int); folds=a.fold_id.to_numpy(int); base=a.v7_5_pred.to_numpy(bool); bm=v75.metrics(y,base,folds)
    configs=[]
    for features,C,tpq,margin,scope,guard,rescue in product(["core","all"],[.05,.1,.25,.5],[0,.1,.2],[.02,.05],["market","nonstrong","any"],[.30,.40,.50],[1.01,.85,.90]):
        configs.append({"features":features,"C":C,"tp_quantile":tpq,"margin":margin,"scope":scope,"rank_guard":guard,"rescue":rescue})
    rows=[]; preds=[]
    for i,cfg in enumerate(configs):
        p,details=candidate(a,b,e,x,strong,quiet,market,cfg); m=v75.metrics(y,p,folds)
        strict=(m["recall"]>=bm["recall"] and m["green_npv"]>=bm["green_npv"] and m["precision"]>bm["precision"] and m["f1"]>bm["f1"] and m["fp"]<bm["fp"] and m["worst_fold_recall"]>=bm["worst_fold_recall"])
        rows.append({"config_id":i,"config":cfg,"metrics":m,"strictly_dominates_v7_5":bool(strict),"details":details}); preds.append(p)
    def key(r):
        m=r["metrics"]; return (int(r["strictly_dominates_v7_5"]),m["f1"],m["precision"],-m["fp"],m["recall"],m["green_npv"],-m["alert_rate"])
    sel=max(rows,key=key); p=preds[sel["config_id"]]; m=sel["metrics"]
    report={"version":VERSION,"status":"DEVELOPMENT_BEST" if sel["strictly_dominates_v7_5"] else "EXPERIMENTAL_V8_2_REMAINS_BEST","scientific_boundary":"V8.5 adds the OOF V7.6 ensemble ranking to a verifier trained only on earlier V7.5 alerts for each fold. Configuration selection remains development selection; external validation is pending.","candidate_count":len(rows),"v7_5":bm,"selected":sel,"top_candidates":sorted(rows,key=key,reverse=True)[:10],"red_supported":False}
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    pd.DataFrame({"date":a.date,"y":y,"fold_id":folds,"v7_5_pred":base.astype(int),"v8_rank":b.v8_rank,"v76_score":e.ensemble_score,"v8_5_pred":p.astype(int)}).to_csv(OOF,index=False)
    lines=["# Sales Sentinel V8.5 — Prequential Consensus Alert Verifier","",f"- Status: **{report['status']}**",f"- Candidates: **{len(rows)}**",f"- Selected: **{sel['config']}**","",f"- Precision: V7.5 **{bm['precision']:.2%}** -> V8.5 **{m['precision']:.2%}**",f"- Recall: V7.5 **{bm['recall']:.2%}** -> V8.5 **{m['recall']:.2%}**",f"- F1: V7.5 **{bm['f1']:.2%}** -> V8.5 **{m['f1']:.2%}**",f"- NPV: V7.5 **{bm['green_npv']:.2%}** -> V8.5 **{m['green_npv']:.2%}**",f"- Alert rate: **{m['alert_rate']:.2%}**",f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",f"- Strictly dominates V7.5: **{sel['strictly_dominates_v7_5']}**","","Scientific boundary: development-selected causal stacking; external validation pending."]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
