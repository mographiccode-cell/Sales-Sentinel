from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
VERSION = "SALES-SENTINEL-V7.5-NOISE-AWARE-PREQUENTIAL-ERROR-CORRECTOR"
SEED = 42
PURGE_DAYS = 7
PANEL = ROOT / "data" / "merchant_v7_1" / "merchant_feature_panel_v7_1.csv"
V61_DIAG = ROOT / "reports" / "merchant_market_fusion_v6_1" / "oof_policy_diagnostics.csv"
V61_REPORT = ROOT / "reports" / "merchant_market_fusion_v6_1" / "development_report.json"
OUT = ROOT / "reports" / "merchant_error_corrector_v7_5"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "development_report.json"
SUMMARY = OUT / "development_summary.md"
OOF = OUT / "oof_candidate_scores.csv"
FUSED = OUT / "oof_fused_predictions.csv"

META = {"date","future_ratio","future7_sales","baseline28_daily","target"}


def prepare(Xtr, Xva):
    tr=Xtr.copy(); va=Xva.copy()
    for col in tr.columns:
        a=pd.to_numeric(tr[col],errors="coerce").replace([np.inf,-np.inf],np.nan)
        f=a.dropna()
        if f.empty: lo=hi=med=0.0
        else: lo=float(f.quantile(.01)); hi=float(f.quantile(.99)); med=float(f.median())
        tr[col]=pd.to_numeric(tr[col],errors="coerce").clip(lo,hi).fillna(med)
        va[col]=pd.to_numeric(va[col],errors="coerce").clip(lo,hi).fillna(med)
    return tr.astype(float),va.astype(float)


def stable_top(X,y,n):
    y=np.asarray(y,float); cut=max(int(len(y)*.55),40)
    scores=[]
    for col in X.columns:
        x=np.asarray(X[col],float)
        s1=abs(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-12 else 0.0
        xr=x[-cut:]; yr=y[-cut:]
        s2=abs(np.corrcoef(xr,yr)[0,1]) if np.std(xr)>1e-12 and np.std(yr)>1e-12 else 0.0
        if not np.isfinite(s1): s1=0.0
        if not np.isfinite(s2): s2=0.0
        scores.append((.65*s1+.35*s2,col))
    scores.sort(reverse=True)
    return [c for _,c in scores[:n]]


def sample_weight(y, future_ratio, enabled):
    y=np.asarray(y,int); r=np.asarray(future_ratio,float)
    if not enabled: return np.ones(len(y),float)
    pos=max(int((y==1).sum()),1); neg=max(int((y==0).sum()),1)
    balance=np.where(y==1,neg/pos,1.0)
    margin=np.abs(r-.85)
    confidence=np.clip(.55+margin/.12,.55,1.65)
    return balance*confidence


def make_model(kind,y):
    y=np.asarray(y,int); pos=max(int((y==1).sum()),1); neg=max(int((y==0).sum()),1); spw=neg/pos
    if kind=="catboost":
        return CatBoostClassifier(iterations=550,depth=4,learning_rate=.022,l2_leaf_reg=14.0,random_seed=SEED,verbose=False,allow_writing_files=False,loss_function="Logloss")
    if kind=="xgb":
        return XGBClassifier(n_estimators=520,max_depth=2,learning_rate=.022,min_child_weight=8,subsample=.86,colsample_bytree=.72,reg_alpha=1.8,reg_lambda=14.0,gamma=.18,objective="binary:logistic",eval_metric="logloss",random_state=SEED,n_jobs=2,scale_pos_weight=1.0)
    raise KeyError(kind)


def percentile(score,ref):
    ref=np.sort(np.asarray(ref,float)); return np.searchsorted(ref,np.asarray(score,float),side="right")/max(len(ref),1)


def windows(d):
    w=[("2023-07-08","2023-09-30"),("2023-10-08","2023-12-31"),("2024-01-08","2024-03-31"),("2024-04-08","2024-06-30"),("2024-07-08","2024-08-19")]
    out=[]
    for fid,(a,b) in enumerate(w):
        a=pd.Timestamp(a); b=pd.Timestamp(b)
        tr=d.date<=a-pd.Timedelta(days=PURGE_DAYS+1); va=d.date.between(a,b)
        out.append((fid,tr,va))
    return out


def candidate_oof(d,config):
    all_features=[c for c in d.columns if c not in META]
    if config["feature_mode"]=="merchant_regime":
        base=[c for c in all_features if c.startswith(("merchant__","market__","calendar__","catregime__"))]
    else:
        base=all_features
    parts=[]; fold_stats=[]
    for fid,tr,va in windows(d):
        Xtr0,Xva0=prepare(d.loc[tr,base],d.loc[va,base]); ytr=d.loc[tr,"target"].astype(int)
        if config["feature_mode"]=="top96":
            cols=stable_top(Xtr0,ytr,96); Xtr=Xtr0[cols]; Xva=Xva0[cols]
        else:
            cols=list(Xtr0.columns); Xtr=Xtr0; Xva=Xva0
        m=make_model(config["model"],ytr)
        sw=sample_weight(ytr,d.loc[tr,"future_ratio"],config["weighted"])
        m.fit(Xtr,ytr,sample_weight=sw)
        train_score=m.predict_proba(Xtr)[:,1]; val_score=m.predict_proba(Xva)[:,1]
        rank=percentile(val_score,train_score); yy=d.loc[va,"target"].astype(int).to_numpy()
        auc=float(roc_auc_score(yy,rank)); pr=float(average_precision_score(yy,rank))
        fold_stats.append({"fold_id":fid,"rows":int(va.sum()),"positives":int(yy.sum()),"roc_auc":auc,"pr_auc":pr,"feature_count":len(cols)})
        parts.append(pd.DataFrame({"date":d.loc[va,"date"].to_numpy(),"target":yy,"fold_id":fid,"rank_score":rank,"raw_score":val_score}))
    o=pd.concat(parts,ignore_index=True)
    y=o.target.to_numpy(int); r=o.rank_score.to_numpy(float)
    return o,{"roc_auc":float(roc_auc_score(y,r)),"pr_auc":float(average_precision_score(y,r)),"min_fold_auc":float(min(x["roc_auc"] for x in fold_stats)),"mean_fold_pr":float(np.mean([x["pr_auc"] for x in fold_stats])),"folds":fold_stats}


def base_components(diag):
    mm=diag.merchant_mean.to_numpy(float); lr=diag.merchant_logreg.to_numpy(float); ex=diag.merchant_extra.to_numpy(float); dis=diag.merchant_disagreement.to_numpy(float); mp=diag.market_v3__risk_p90.to_numpy(float)
    strong=(mm>=.70)&(ex>=.60)
    quiet=(mp<=.05)&(lr>=.45)&(dis>=.10)&(mm>=.35)
    market=(mp>=.20)&(mm>=.35)
    return strong,quiet,market,strong|quiet|market


def metrics(y,pred,folds):
    y=np.asarray(y,int); p=np.asarray(pred,bool); folds=np.asarray(folds,int)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&(~p)).sum()); tn=int(((y==0)&(~p)).sum())
    per=[]
    for f in sorted(np.unique(folds)):
        ix=folds==f; yy=y[ix]; pp=p[ix]
        ftp=int(((yy==1)&pp).sum()); ffp=int(((yy==0)&pp).sum()); ffn=int(((yy==1)&(~pp)).sum()); ftn=int(((yy==0)&(~pp)).sum())
        per.append({"fold_id":int(f),"recall":float(ftp/max(ftp+ffn,1)),"precision":float(ftp/max(ftp+ffp,1)),"f1":float(2*ftp/max(2*ftp+ffp+ffn,1)),"alert_rate":float(pp.mean()),"green_npv":float(ftn/max(ftn+ffn,1)),"tp":ftp,"fp":ffp,"fn":ffn,"tn":ftn})
    return {"precision":float(tp/max(tp+fp,1)),"recall":float(tp/max(tp+fn,1)),"f1":float(2*tp/max(2*tp+fp+fn,1)),"accuracy":float((tp+tn)/len(y)),"balanced_accuracy":float(.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1))),"alert_rate":float(p.mean()),"green_npv":float(tn/max(tn+fn,1)),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"worst_fold_recall":float(min(x["recall"] for x in per)),"max_fold_alert_rate":float(max(x["alert_rate"] for x in per)),"per_fold":per}


def apply_rule(base,strong,quiet,market,score,mm,mp,rule):
    pred=base.copy()
    scope,qv,rscope,qr=rule
    if scope!="none":
        if scope=="quiet": mask=quiet&(~strong)&(~market)
        elif scope=="market": mask=market&(~strong)
        elif scope=="nonstrong": mask=~strong
        else: mask=np.ones(len(base),bool)
        pred=pred & ~(base&mask&(score<qv))
    if rscope!="none":
        if rscope=="near": rmask=(mm>=.28)
        elif rscope=="market": rmask=(mp>=.10)
        else: rmask=np.ones(len(base),bool)
        pred=pred | ((~base)&rmask&(score>=qr))
    return pred


def rule_grid():
    scopes=["none","quiet","market","nonstrong","any"]
    qvs=[0.0,.10,.20,.30,.40,.50]
    rscopes=["none","near","market","any"]
    qrs=[1.01,.80,.85,.90,.93,.96]
    out=[]
    for s,qv,rs,qr in product(scopes,qvs,rscopes,qrs):
        if s=="none" and qv!=0.0: continue
        if rs=="none" and qr!=1.01: continue
        if s!="none" and qv==0.0: continue
        if rs!="none" and qr>1.0: continue
        out.append((s,qv,rs,qr))
    return out


def prequential_fusion(y,folds,base,strong,quiet,market,score,mm,mp):
    final=base.copy(); details=[]; rules=rule_grid()
    for f in sorted(np.unique(folds)):
        cur=folds==f
        if f==0:
            details.append({"fold_id":int(f),"mode":"v6_1_bootstrap"}); continue
        hist=folds<f
        bm=metrics(y[hist],base[hist],folds[hist])
        best=None
        for rule in rules:
            pp=apply_rule(base[hist],strong[hist],quiet[hist],market[hist],score[hist],mm[hist],mp[hist],rule)
            m=metrics(y[hist],pp,folds[hist])
            feasible=(m["recall"]>=bm["recall"]-.02 and m["green_npv"]>=bm["green_npv"]-.005 and m["alert_rate"]<=bm["alert_rate"]+.01 and m["fp"]<=bm["fp"] and m["f1"]>=bm["f1"])
            obj=(int(feasible),m["f1"],m["precision"],-m["fp"],m["recall"],-m["alert_rate"])
            if best is None or obj>best[0]: best=(obj,rule,m,feasible)
        _,rule,hm,feasible=best
        if not feasible: rule=("none",0.0,"none",1.01)
        final[cur]=apply_rule(base[cur],strong[cur],quiet[cur],market[cur],score[cur],mm[cur],mp[cur],rule)
        details.append({"fold_id":int(f),"history_rule":list(rule),"history_feasible":bool(feasible),"history_metrics":hm})
    return final,details


def oracle_ceiling(y,folds,base,strong,quiet,market,score,mm,mp,base_metrics):
    best=None
    for rule in rule_grid():
        p=apply_rule(base,strong,quiet,market,score,mm,mp,rule); m=metrics(y,p,folds)
        feasible=(m["recall"]>=.80 and m["green_npv"]>=.95 and m["alert_rate"]<=base_metrics["alert_rate"] and m["worst_fold_recall"]>=.60 and m["fp"]<base_metrics["fp"] and m["f1"]>base_metrics["f1"])
        key=(int(feasible),m["f1"],m["precision"],-m["fp"],m["recall"])
        if best is None or key>best[0]: best=(key,rule,m,feasible)
    return {"rule":list(best[1]),"metrics":best[2],"strict_feasible":bool(best[3])}


def main():
    d=pd.read_csv(PANEL,parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    diag=pd.read_csv(V61_DIAG)
    strong,quiet,market,base=base_components(diag)
    y=diag.y.to_numpy(int); folds=diag.fold_id.to_numpy(int); mm=diag.merchant_mean.to_numpy(float); mp=diag.market_v3__risk_p90.to_numpy(float)
    bm=metrics(y,base,folds)
    report_ref=json.loads(V61_REPORT.read_text(encoding="utf-8"))["selected_policy"]["metrics"]
    if (bm["tp"],bm["fp"],bm["fn"],bm["tn"])!=(52,82,11,236): raise RuntimeError("V6.1 reconstruction mismatch")

    configs=[]
    for model in ["catboost","xgb"]:
        for feature_mode in ["all","merchant_regime","top96"]:
            for weighted in [True,False]:
                if not weighted and feature_mode!="all": continue
                configs.append({"model":model,"feature_mode":feature_mode,"weighted":weighted})

    candidates=[]; all_oof=[]
    for i,cfg in enumerate(configs):
        o,rankm=candidate_oof(d,cfg)
        o=o.sort_values(["fold_id","date"]).reset_index(drop=True)
        # Match V6.1 by fold and chronological position; exact target sequence is mandatory.
        if len(o)!=len(diag) or not np.array_equal(o.target.to_numpy(int),y) or not np.array_equal(o.fold_id.to_numpy(int),folds):
            raise RuntimeError(f"OOF alignment mismatch for {cfg}")
        score=o.rank_score.to_numpy(float)
        fused,details=prequential_fusion(y,folds,base,strong,quiet,market,score,mm,mp)
        fm=metrics(y,fused,folds)
        oracle=oracle_ceiling(y,folds,base,strong,quiet,market,score,mm,mp,bm)
        strict=(fm["recall"]>=.80 and fm["precision"]>bm["precision"] and fm["f1"]>bm["f1"] and fm["green_npv"]>=.95 and fm["alert_rate"]<=bm["alert_rate"] and fm["worst_fold_recall"]>=.60 and fm["fp"]<bm["fp"])
        row={"config_id":i,"config":cfg,"ranking":rankm,"prequential_fusion":fm,"prequential_details":details,"strict_adoptable":bool(strict),"oracle_development_ceiling":oracle}
        candidates.append(row)
        oo=o.copy(); oo["config_id"]=i; all_oof.append(oo)
        print(json.dumps({"config":cfg,"ranking":rankm,"fusion":fm,"strict":strict,"oracle":oracle["strict_feasible"]}))

    def key(r):
        m=r["prequential_fusion"]; rk=r["ranking"]
        return (int(r["strict_adoptable"]),m["f1"],m["precision"],m["recall"],rk["min_fold_auc"],rk["pr_auc"],-m["fp"])
    selected=max(candidates,key=key); sm=selected["prequential_fusion"]
    adopt=bool(selected["strict_adoptable"])
    pd.concat(all_oof,ignore_index=True).to_csv(OOF,index=False)

    sel_oof=pd.concat(all_oof,ignore_index=True)
    sel_oof=sel_oof[sel_oof.config_id==selected["config_id"]].sort_values(["fold_id","date"]).reset_index(drop=True)
    score=sel_oof.rank_score.to_numpy(float)
    pred,details=prequential_fusion(y,folds,base,strong,quiet,market,score,mm,mp)
    pd.DataFrame({"date":sel_oof.date,"y":y,"fold_id":folds,"v6_1_pred":base.astype(int),"v7_5_rank":score,"v7_5_pred":pred.astype(int)}).to_csv(FUSED,index=False)

    report={"version":VERSION,"status":"DEVELOPMENT_ADOPTABLE_PENDING_EXTERNAL_VALIDATION" if adopt else "EXPERIMENTAL_NOT_ADOPTED","scientific_boundary":"V7.5 trains noise-aware ranking candidates on the existing 541-row merchant panel and applies only prequential error-correction rules to frozen V6.1 decisions. Candidate/model selection is development evidence on previously used folds; no fresh external Saudi merchant validation has occurred.","rows":len(d),"oof_rows":len(y),"candidate_count":len(candidates),"v6_1_base":bm,"v6_1_report_reference":report_ref,"selected":selected,"all_candidates":candidates,"adopt_over_v6_1":adopt}
    REPORT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    rk=selected["ranking"]
    summary=["# Sales Sentinel V7.5 — Noise-Aware Prequential Error Corrector","",f"- Status: **{report['status']}**",f"- Selected candidate: **{selected['config']}**",f"- Candidate ROC-AUC / PR-AUC: **{rk['roc_auc']:.2%} / {rk['pr_auc']:.2%}**",f"- Minimum fold ROC-AUC: **{rk['min_fold_auc']:.2%}**","",f"- V6.1 precision / recall / F1: **{bm['precision']:.2%} / {bm['recall']:.2%} / {bm['f1']:.2%}**",f"- V7.5 precision / recall / F1: **{sm['precision']:.2%} / {sm['recall']:.2%} / {sm['f1']:.2%}**",f"- V6.1 NPV / alert rate: **{bm['green_npv']:.2%} / {bm['alert_rate']:.2%}**",f"- V7.5 NPV / alert rate: **{sm['green_npv']:.2%} / {sm['alert_rate']:.2%}**",f"- V6.1 TP/FP/FN/TN: **{bm['tp']}/{bm['fp']}/{bm['fn']}/{bm['tn']}**",f"- V7.5 TP/FP/FN/TN: **{sm['tp']}/{sm['fp']}/{sm['fn']}/{sm['tn']}**",f"- Worst-fold recall: **{sm['worst_fold_recall']:.2%}**",f"- Oracle development ceiling can strictly dominate V6.1: **{selected['oracle_development_ceiling']['strict_feasible']}**",f"- Adopt over V6.1: **{adopt}**","- RED supported: **False**","","Scientific boundary: development-only evidence; external real Saudi merchant validation remains required."]
    SUMMARY.write_text("\n".join(summary)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))

if __name__=="__main__": main()
