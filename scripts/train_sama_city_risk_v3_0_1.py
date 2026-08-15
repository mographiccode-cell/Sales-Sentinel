from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

import train_sama_city_risk_v3 as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'sama_city_v3_0_1'; MOD=ROOT/'models'/'sama_city_v3_0_1'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
MODEL=MOD/'city_risk_v3_0_1.joblib'; REPORT=OUT/'development_report.json'
VERSION='SAMA-CITY-RISK-3.0.1-GENERALIZATION-FIRST'


def bin_metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,bool)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&~p).sum()); tn=int(((y==0)&~p).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}


def valid_watch(q,t):
    y=q.y.to_numpy(int); alert=(q.score.to_numpy(float)>=t)|(q.precursor_count.to_numpy(int)>=2)
    m=bin_metrics(y,alert); per=[]
    for fid,z in q.groupby('fold_id'):
        yy=z.y.to_numpy(int); aa=(z.score.to_numpy(float)>=t)|(z.precursor_count.to_numpy(int)>=2)
        mm=bin_metrics(yy,aa); per.append({'fold_id':int(fid),'positives':int(yy.sum()),'recall':mm['recall'],'NPV':mm['NPV']})
    pf=[x for x in per if x['positives']>=2]
    worst=min((x['recall'] for x in pf),default=1.0)
    ok=m['recall']>=base.CONTRACT['pooled_alert_recall_min'] and m['NPV']>=base.CONTRACT['pooled_green_npv_min'] and worst>=base.CONTRACT['worst_positive_fold_alert_recall_min']
    return ok,m,per,worst


def valid_red(q,t):
    y=q.y.to_numpy(int); red=(q.score.to_numpy(float)>=t)&(q.agreement.to_numpy(int)>=2)&(q.precursor_count.to_numpy(int)>=2)
    m=bin_metrics(y,red); per=[]
    for fid,z in q.groupby('fold_id'):
        yy=z.y.to_numpy(int); rr=(z.score.to_numpy(float)>=t)&(z.agreement.to_numpy(int)>=2)&(z.precursor_count.to_numpy(int)>=2)
        mm=bin_metrics(yy,rr); per.append({'fold_id':int(fid),'FPR':mm['FPR'],'precision':mm['precision'],'alerts':int(rr.sum())})
    worst=max((x['FPR'] for x in per),default=0.0); n=int(red.sum())
    ok=n>=base.CONTRACT['min_red_alerts'] and m['precision']>=base.CONTRACT['pooled_red_precision_min'] and m['FPR']<=base.CONTRACT['pooled_red_fpr_max'] and worst<=base.CONTRACT['worst_fold_red_fpr_max']
    return ok,m,per,worst,n


def choose_fast(q):
    scores=q.score.to_numpy(float)
    cand=np.unique(np.r_[np.quantile(scores,np.linspace(.02,.995,90)),np.linspace(.02,.85,70)])
    watches=[]; reds=[]
    for t in cand:
        ok,m,per,worst=valid_watch(q,float(t))
        if ok: watches.append((float(t),m,per,worst))
        ok,m,per,worst,n=valid_red(q,float(t))
        if ok: reds.append((float(t),m,per,worst,n))
    if not watches: raise RuntimeError('No v3.0.1 WATCH threshold meets cross-regime contract')
    if not reds: raise RuntimeError('No v3.0.1 RED threshold meets cross-regime contract')
    best=None
    for w in watches:
        for r in reds:
            if r[0]<w[0]: continue
            obj=(r[1]['precision'],w[1]['recall'],-r[1]['FP'],r[0],w[0])
            if best is None or obj>best[0]: best=(obj,w,r)
    if best is None: raise RuntimeError('No compatible WATCH/RED threshold pair')
    return best[1],best[2],{'watch_candidates':len(watches),'red_candidates':len(reds),'grid_candidates':len(cand)}


def main():
    panel=base.source.reconciled_load_panel(base.HISTORY)
    d,X,P,pc=base.featureize(panel,require_target=True)
    keep=d.week_start<=base.DEV_END; d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True); pc=pc.loc[keep].reset_index(drop=True)
    forbidden=[c for c in X.columns if c.startswith('city_') or 'decline_rate' in c or 'target' in c or 'future' in c]
    if forbidden: raise RuntimeError(f'Forbidden features: {forbidden}')
    oof,folds=base.build_oof(d,X,pc)
    roc=float(roc_auc_score(oof.y,oof.score)); pr=float(average_precision_score(oof.y,oof.score))
    w,r,search=choose_fast(oof); wt,wm,wper,wworst=w; rt,rm,rper,rworst,rn=r
    gates={
        'roc_auc':roc>=base.CONTRACT['pooled_roc_auc_min'],
        'red_precision':rm['precision']>=base.CONTRACT['pooled_red_precision_min'],
        'red_fpr':rm['FPR']<=base.CONTRACT['pooled_red_fpr_max'],
        'alert_recall':wm['recall']>=base.CONTRACT['pooled_alert_recall_min'],
        'green_npv':wm['NPV']>=base.CONTRACT['pooled_green_npv_min'],
        'worst_positive_fold_recall':wworst>=base.CONTRACT['worst_positive_fold_alert_recall_min'],
        'worst_fold_red_fpr':rworst<=base.CONTRACT['worst_fold_red_fpr_max'],
    }
    fitted={}
    for name,factory in base.model_factories().items(): fitted[name]=base.fit_one(clone(factory),X,d.target)
    artifact={
        'version':VERSION,'models':fitted,'features':list(X.columns),'watch_threshold':wt,'red_threshold':rt,'min_precursor_red':2,
        'ood_profile':base.robust_ood_profile(X),'ood_max_fraction':.15,'development_end':str(base.DEV_END.date()),
        'target_definition':'next-week city POS value <80% current trailing-4-week mean','scope':'forecastable deterioration; surprise shocks without leading information are outside predictive claim',
    }
    joblib.dump(artifact,MODEL)
    report={
        'version':VERSION,'rows':int(len(d)),'positives':int(d.target.sum()),'positive_rate':float(d.target.mean()),'feature_count':len(X.columns),
        'folds':folds,'ranking':{'ROC_AUC':roc,'PR_AUC':pr},'threshold_search':search,
        'policy':{'watch_threshold':wt,'red_threshold':rt,'RED':rm,'RED_plus_AMBER':wm,'GREEN':{'NPV':wm['NPV'],'missed_declines':wm['FN']},'worst_positive_fold_alert_recall':wworst,'worst_fold_red_fpr':rworst,'red_alerts':rn},
        'watch_fold_metrics':wper,'red_fold_metrics':rper,'contract':base.CONTRACT,'gates':gates,'all_gates_passed':bool(all(gates.values())),
        'controls':{'no_city_identity':True,'no_target_prevalence_features':True,'no_absolute_levels':True,'multi_model_consensus':True,'precursor_gated_red':True,'ood_profile':True,'purged_multi_regime_oof':True,'thresholds_selected_across_all_oof_regimes':True},
        'scientific_boundary':'No post-2025-06-29 outcome is used for model or threshold fitting.'
    }
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'development_summary.md').write_text(
        '# Sales Sentinel v3.0.1\n\n'
        f"- Features: **{len(X.columns)}**\n- OOF ROC-AUC: **{roc:.2%}**; PR-AUC: **{pr:.2%}**\n"
        f"- RED precision: **{rm['precision']:.2%}** ({rm['TP']} TP / {rm['FP']} FP); FPR **{rm['FPR']:.2%}**\n"
        f"- RED+AMBER recall: **{wm['recall']:.2%}**; GREEN NPV **{wm['NPV']:.2%}**\n"
        f"- Worst positive-fold alert recall: **{wworst:.2%}**; worst fold RED FPR **{rworst:.2%}**\n"
        f"- Gates: **{report['all_gates_passed']}**\n",encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
