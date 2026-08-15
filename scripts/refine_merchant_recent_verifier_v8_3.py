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
VERSION="SALES-SENTINEL-V8.3-RECENT-WINDOW-ALERT-VERIFIER"
OUT=ROOT/"reports"/"merchant_recent_verifier_v8_3"; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/"development_report.json"; SUMMARY=OUT/"development_summary.md"; OOF=OUT/"oof_predictions.csv"


def fit_predict(Xtr,ytr,Xva,C):
    sc=StandardScaler(); A=sc.fit_transform(Xtr); B=sc.transform(Xva)
    m=LogisticRegression(C=C,class_weight="balanced",max_iter=2000,solver="liblinear",random_state=42)
    m.fit(A,ytr); return m.predict_proba(A)[:,1],m.predict_proba(B)[:,1]


def causal(a,b,x,strong,quiet,market,cfg):
    y=a.y.to_numpy(int); folds=a.fold_id.to_numpy(int); base=a.v7_5_pred.to_numpy(bool); score=b.v8_rank.to_numpy(float)
    final=base.copy(); details=[]
    core=[c for c in x.columns if not c.startswith("v8__")]
    for f in sorted(np.unique(folds)):
        cur=folds==f
        if f==0:
            details.append({"fold_id":0,"mode":"bootstrap"}); continue
        low=max(0,f-cfg["history_folds"])
        hist=(folds<f)&(folds>=low); ha=hist&base
        if ha.sum()<20 or len(np.unique(y[ha]))<2:
            details.append({"fold_id":int(f),"mode":"insufficient_history","history_folds":[int(low),int(f-1)]}); continue
        ptr,pcur=fit_predict(x.loc[ha,core],y[ha],x.loc[cur,core],cfg["C"])
        tp=ptr[y[ha]==1]
        thr=float(max(0,np.quantile(tp,cfg["tp_quantile"])-cfg["margin"]))
        if cfg["scope"]=="market": scope=market[cur]&(~strong[cur])
        elif cfg["scope"]=="nonstrong": scope=~strong[cur]
        else: scope=np.ones(cur.sum(),bool)
        cb=base[cur].copy(); veto=cb&scope&(pcur<thr)&(score[cur]<cfg["rank_guard"])
        cp=cb&(~veto)
        if cfg["rescue"]<=1:
            cp=cp|((~cb)&market[cur]&(score[cur]>=cfg["rescue"]))
        final[cur]=cp
        details.append({"fold_id":int(f),"history_folds":[int(low),int(f-1)],"history_alerts":int(ha.sum()),"history_tp":int(y[ha].sum()),"threshold":thr,"vetoes":int(veto.sum())})
    return final,details


def main():
    a,b,d,x,strong,quiet,market=v82.build_oof_features(); y=a.y.to_numpy(int); folds=a.fold_id.to_numpy(int); base=a.v7_5_pred.to_numpy(bool); bm=v75.metrics(y,base,folds)
    configs=[]
    for hf,C,tpq,margin,scope,guard,rescue in product([1,2,3,5],[.05,.1,.25],[0,.1,.2],[.02,.05,.08],["market","nonstrong"],[.30,.35,.40,.45,.50],[1.01,.85,.90]):
        configs.append({"history_folds":hf,"C":C,"tp_quantile":tpq,"margin":margin,"scope":scope,"rank_guard":guard,"rescue":rescue})
    rows=[]; preds=[]
    for i,cfg in enumerate(configs):
        p,details=causal(a,b,x,strong,quiet,market,cfg); m=v75.metrics(y,p,folds)
        strict=(m["recall"]>=bm["recall"] and m["green_npv"]>=bm["green_npv"] and m["precision"]>bm["precision"] and m["f1"]>bm["f1"] and m["fp"]<bm["fp"] and m["worst_fold_recall"]>=bm["worst_fold_recall"])
        rows.append({"config_id":i,"config":cfg,"metrics":m,"strictly_dominates_v7_5":bool(strict),"details":details}); preds.append(p)
    def key(r):
        m=r["metrics"]; return (int(r["strictly_dominates_v7_5"]),m["f1"],m["precision"],-m["fp"],m["recall"],m["green_npv"],-m["alert_rate"])
    sel=max(rows,key=key); p=preds[sel["config_id"]]; m=sel["metrics"]
    report={"version":VERSION,"status":"DEVELOPMENT_BEST" if sel["strictly_dominates_v7_5"] else "EXPERIMENTAL_V8_2_RECOMMENDED","scientific_boundary":"V8.3 tests recent-fold verifier memory to adapt to regime drift. Each outer fold still uses only earlier folds, but configuration selection is development selection on existing OOF. Fresh external validation remains required.","candidate_count":len(rows),"v7_5":bm,"selected":sel,"top_candidates":sorted(rows,key=key,reverse=True)[:10],"red_supported":False}
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    pd.DataFrame({"date":a.date,"y":y,"fold_id":folds,"v7_5_pred":base.astype(int),"v8_rank":b.v8_rank,"v8_3_pred":p.astype(int)}).to_csv(OOF,index=False)
    lines=["# Sales Sentinel V8.3 — Recent-Window Alert Verifier","",f"- Status: **{report['status']}**",f"- Candidates: **{len(rows)}**",f"- Selected: **{sel['config']}**","",f"- Precision: V7.5 **{bm['precision']:.2%}** -> V8.3 **{m['precision']:.2%}**",f"- Recall: V7.5 **{bm['recall']:.2%}** -> V8.3 **{m['recall']:.2%}**",f"- F1: V7.5 **{bm['f1']:.2%}** -> V8.3 **{m['f1']:.2%}**",f"- NPV: V7.5 **{bm['green_npv']:.2%}** -> V8.3 **{m['green_npv']:.2%}**",f"- Alert rate: **{m['alert_rate']:.2%}**",f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",f"- Strictly dominates V7.5: **{sel['strictly_dominates_v7_5']}**","","Scientific boundary: causal within folds, development-selected configuration; external validation pending."]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
