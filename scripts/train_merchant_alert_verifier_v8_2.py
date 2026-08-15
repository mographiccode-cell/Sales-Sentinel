from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import train_merchant_error_corrector_v7_5 as v75
import train_merchant_target_refinement_v8 as v8

ROOT=Path(__file__).resolve().parents[1]
VERSION="SALES-SENTINEL-V8.2-PREQUENTIAL-ALERT-VERIFIER"
V75=ROOT/"reports"/"merchant_error_corrector_v7_5"/"oof_fused_predictions.csv"
V8=ROOT/"reports"/"merchant_target_refinement_v8"/"oof_fused_predictions.csv"
DIAG=ROOT/"reports"/"merchant_market_fusion_v6_1"/"oof_policy_diagnostics.csv"
PANEL=ROOT/"data"/"merchant_v7_1"/"merchant_feature_panel_v7_1.csv"
OUT=ROOT/"reports"/"merchant_alert_verifier_v8_2"; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/"development_report.json"; SUMMARY=OUT/"development_summary.md"; OOF=OUT/"oof_predictions.csv"


def build_oof_features():
    a=pd.read_csv(V75).sort_values(["fold_id","date"]).reset_index(drop=True)
    b=pd.read_csv(V8).sort_values(["fold_id","date"]).reset_index(drop=True)
    d=pd.read_csv(DIAG).reset_index(drop=True)
    p=pd.read_csv(PANEL,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    p=v8.add_hard_negative_features(p)
    pp=[]
    for fid,_,va in v75.windows(p):
        q=p.loc[va].copy(); q["fold_id"]=fid; pp.append(q)
    po=pd.concat(pp,ignore_index=True).sort_values(["fold_id","date"]).reset_index(drop=True)
    if not (len(a)==len(b)==len(d)==len(po)==381): raise RuntimeError("length mismatch")
    if not np.array_equal(a.y.to_numpy(int),b.y.to_numpy(int)): raise RuntimeError("target mismatch")
    x=pd.DataFrame(index=a.index)
    x["v75_rank"]=a.v7_5_rank.astype(float)
    x["v8_rank"]=b.v8_rank.astype(float)
    for c in ["merchant_logreg","merchant_extra","merchant_mean","merchant_disagreement","market_v3__risk_mean","market_v3__risk_max","market_v3__risk_p90","market_v3__risk_share_25","market_v3__precursor_mean"]:
        if c in d.columns: x[c]=pd.to_numeric(d[c],errors="coerce").fillna(0.0)
    strong,quiet,market,_=v75.base_components(d)
    x["branch_strong"]=strong.astype(float); x["branch_quiet"]=quiet.astype(float); x["branch_market"]=market.astype(float)
    for c in [c for c in po.columns if c.startswith("v8__")]: x[c]=pd.to_numeric(po[c],errors="coerce").fillna(0.0).to_numpy()
    return a,b,d,x,strong,quiet,market


def fit_predict(Xtr,ytr,Xva,C):
    sc=StandardScaler(); A=sc.fit_transform(Xtr); B=sc.transform(Xva)
    m=LogisticRegression(C=C,class_weight="balanced",max_iter=2000,solver="liblinear",random_state=42)
    m.fit(A,ytr); return m.predict_proba(A)[:,1],m.predict_proba(B)[:,1]


def causal_candidate(a,b,x,strong,quiet,market,cfg):
    y=a.y.to_numpy(int); folds=a.fold_id.to_numpy(int); base=a.v7_5_pred.to_numpy(bool); score=b.v8_rank.to_numpy(float)
    final=base.copy(); details=[]
    core=[c for c in x.columns if not c.startswith("v8__")]
    hard=[c for c in x.columns if c.startswith("v8__")]
    cols=core if cfg["features"]=="core" else core+hard
    for f in sorted(np.unique(folds)):
        cur=folds==f
        if f==0:
            details.append({"fold_id":int(f),"mode":"v7_5_bootstrap"}); continue
        hist=folds<f; ha=hist&base
        if ha.sum()<20 or len(np.unique(y[ha]))<2:
            details.append({"fold_id":int(f),"mode":"insufficient_history"}); continue
        ptr,pcur=fit_predict(x.loc[ha,cols],y[ha],x.loc[cur,cols],cfg["C"])
        tp_probs=ptr[y[ha]==1]
        if len(tp_probs)==0:
            details.append({"fold_id":int(f),"mode":"no_historical_tp"}); continue
        thr=float(max(0.0,np.quantile(tp_probs,cfg["tp_quantile"])-cfg["margin"]))
        if cfg["scope"]=="market": scope=market[cur]&(~strong[cur])
        elif cfg["scope"]=="nonstrong": scope=~strong[cur]
        else: scope=np.ones(cur.sum(),bool)
        cur_base=base[cur].copy()
        # Verifier may only veto an existing alert; rank guard makes the veto conservative.
        veto=cur_base & scope & (pcur<thr) & (score[cur]<cfg["rank_guard"])
        cur_pred=cur_base & (~veto)
        # Optional high-rank rescue is restricted to market evidence.
        if cfg["rescue"]<=1.0:
            rescue=(~cur_base)&(market[cur])&(score[cur]>=cfg["rescue"])
            cur_pred=cur_pred|rescue
        final[cur]=cur_pred
        details.append({"fold_id":int(f),"history_alerts":int(ha.sum()),"history_tp":int(y[ha].sum()),"verifier_threshold":thr,"vetoes":int(veto.sum()),"mode":"verified"})
    return final,details


def main():
    a,b,d,x,strong,quiet,market=build_oof_features(); y=a.y.to_numpy(int); folds=a.fold_id.to_numpy(int); base=a.v7_5_pred.to_numpy(bool)
    bm=v75.metrics(y,base,folds)
    configs=[]
    for features,C,tpq,margin,scope,guard,rescue in product(["core","hard"],[.1,.5,1.0,2.0],[0.0,.10],[.02,.05],["market","nonstrong","any"],[.30,.40,.50],[1.01,.80,.85,.90]):
        configs.append({"features":features,"C":C,"tp_quantile":tpq,"margin":margin,"scope":scope,"rank_guard":guard,"rescue":rescue})
    rows=[]; preds=[]
    for i,cfg in enumerate(configs):
        p,details=causal_candidate(a,b,x,strong,quiet,market,cfg); m=v75.metrics(y,p,folds)
        adopt=(m["recall"]>=bm["recall"] and m["green_npv"]>=bm["green_npv"] and m["precision"]>bm["precision"] and m["f1"]>bm["f1"] and m["fp"]<bm["fp"] and m["worst_fold_recall"]>=bm["worst_fold_recall"])
        rows.append({"config_id":i,"config":cfg,"metrics":m,"strictly_dominates_v7_5":bool(adopt),"details":details}); preds.append(p)
    def key(r):
        m=r["metrics"]; return (int(r["strictly_dominates_v7_5"]),m["f1"],m["precision"],-m["fp"],m["recall"],m["green_npv"],-m["alert_rate"])
    sel=max(rows,key=key); p=preds[sel["config_id"]]; m=sel["metrics"]
    report={"version":VERSION,"status":"DEVELOPMENT_BEST" if sel["strictly_dominates_v7_5"] else "EXPERIMENTAL_V7_5_REMAINS_BEST","scientific_boundary":"Each V8.2 outer fold is modified only by an alert-verifier fitted to earlier V7.5 OOF alerts. Hyperparameter selection across these causal candidates is still development selection; fresh Saudi merchant validation remains required.","candidate_count":len(rows),"v7_5":bm,"selected":sel,"top_candidates":sorted(rows,key=key,reverse=True)[:10],"red_supported":False}
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    pd.DataFrame({"date":a.date,"y":y,"fold_id":folds,"v7_5_pred":base.astype(int),"v8_rank":b.v8_rank,"v8_2_pred":p.astype(int)}).to_csv(OOF,index=False)
    lines=["# Sales Sentinel V8.2 — Prequential Alert Verifier","",f"- Status: **{report['status']}**",f"- Candidates: **{len(rows)}**",f"- Selected: **{sel['config']}**","",f"- Precision: V7.5 **{bm['precision']:.2%}** -> V8.2 **{m['precision']:.2%}**",f"- Recall: V7.5 **{bm['recall']:.2%}** -> V8.2 **{m['recall']:.2%}**",f"- F1: V7.5 **{bm['f1']:.2%}** -> V8.2 **{m['f1']:.2%}**",f"- NPV: V7.5 **{bm['green_npv']:.2%}** -> V8.2 **{m['green_npv']:.2%}**",f"- Alert rate: V7.5 **{bm['alert_rate']:.2%}** -> V8.2 **{m['alert_rate']:.2%}**",f"- TP/FP/FN/TN: **{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}**",f"- Worst-fold recall: **{m['worst_fold_recall']:.2%}**",f"- Strictly dominates V7.5: **{sel['strictly_dominates_v7_5']}**","","Scientific boundary: development-selected causal verifier; external validation remains pending."]
    SUMMARY.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
