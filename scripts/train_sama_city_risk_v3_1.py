from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

import train_sama_city_risk_v3 as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'sama_city_v3_1'; MOD=ROOT/'models'/'sama_city_v3_1'
OUT.mkdir(parents=True,exist_ok=True); MOD.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'development_report.json'; MODEL=MOD/'city_risk_v3_1.joblib'
VERSION='SAMA-CITY-RISK-3.1-SELECTIVE-TRIAGE'
CONTRACT={
    'roc_auc_min':.85,'pr_auc_min':.45,
    'red_precision_min':.70,'red_fpr_max':.015,'min_red_alerts':8,'worst_fold_red_fpr_max':.04,
    'alert_recall_min':.85,'green_npv_min':.985,'alert_precision_min':.15,
    'alert_rate_max':.35,'green_coverage_min':.65,'worst_positive_fold_alert_recall_min':.50,
}


def bm(y,p):
    y=np.asarray(y,int); p=np.asarray(p,bool)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&~p).sum()); tn=int(((y==0)&~p).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}


def eval_watch(q,t):
    alert=q.score.to_numpy(float)>=t; y=q.y.to_numpy(int); m=bm(y,alert)
    rate=float(alert.mean()); green=1-rate; per=[]
    for fid,z in q.groupby('fold_id'):
        yy=z.y.to_numpy(int); aa=z.score.to_numpy(float)>=t; mm=bm(yy,aa)
        per.append({'fold_id':int(fid),'positives':int(yy.sum()),'recall':mm['recall'],'precision':mm['precision'],'alert_rate':float(aa.mean()),'NPV':mm['NPV']})
    pf=[x for x in per if x['positives']>=2]; worst=min((x['recall'] for x in pf),default=1.0)
    ok=(m['recall']>=CONTRACT['alert_recall_min'] and m['NPV']>=CONTRACT['green_npv_min'] and m['precision']>=CONTRACT['alert_precision_min'] and rate<=CONTRACT['alert_rate_max'] and green>=CONTRACT['green_coverage_min'] and worst>=CONTRACT['worst_positive_fold_alert_recall_min'])
    return ok,m,rate,green,worst,per


def eval_red(q,t):
    red=(q.score.to_numpy(float)>=t)&(q.agreement.to_numpy(int)>=2)&(q.precursor_count.to_numpy(int)>=2); y=q.y.to_numpy(int); m=bm(y,red)
    per=[]
    for fid,z in q.groupby('fold_id'):
        yy=z.y.to_numpy(int); rr=(z.score.to_numpy(float)>=t)&(z.agreement.to_numpy(int)>=2)&(z.precursor_count.to_numpy(int)>=2); mm=bm(yy,rr)
        per.append({'fold_id':int(fid),'FPR':mm['FPR'],'precision':mm['precision'],'alerts':int(rr.sum())})
    worst=max((x['FPR'] for x in per),default=0.0); n=int(red.sum())
    ok=(n>=CONTRACT['min_red_alerts'] and m['precision']>=CONTRACT['red_precision_min'] and m['FPR']<=CONTRACT['red_fpr_max'] and worst<=CONTRACT['worst_fold_red_fpr_max'])
    return ok,m,n,worst,per


def choose(q):
    scores=q.score.to_numpy(float)
    cand=np.unique(np.r_[np.quantile(scores,np.linspace(.03,.995,120)),np.linspace(.02,.90,90)])
    watches=[]; reds=[]
    for t in cand:
        a=eval_watch(q,float(t))
        if a[0]: watches.append((float(t),)+a[1:])
        r=eval_red(q,float(t))
        if r[0]: reds.append((float(t),)+r[1:])
    if not watches: raise RuntimeError('No selective WATCH policy meets v3.1 operational contract')
    if not reds: raise RuntimeError('No selective RED policy meets v3.1 operational contract')
    best=None
    for w in watches:
        for r in reds:
            if r[0]<w[0]: continue
            # w=(t,m,rate,green,worst,per), r=(t,m,n,worst,per)
            obj=(r[1]['precision'],w[1]['precision'],w[1]['recall'],w[3],-r[1]['FP'],r[0])
            if best is None or obj>best[0]: best=(obj,w,r)
    if best is None: raise RuntimeError('No compatible v3.1 RED/WATCH pair')
    return best[1],best[2],{'watch_candidates':len(watches),'red_candidates':len(reds),'total_candidates':len(cand)}


def main():
    panel=base.source.reconciled_load_panel(base.HISTORY)
    d,X,P,pc=base.featureize(panel,require_target=True); keep=d.week_start<=base.DEV_END
    d=d.loc[keep].reset_index(drop=True); X=X.loc[keep].reset_index(drop=True); pc=pc.loc[keep].reset_index(drop=True)
    forbidden=[c for c in X.columns if c.startswith('city_') or 'decline_rate' in c or 'target' in c or 'future' in c]
    if forbidden: raise RuntimeError(f'Forbidden features: {forbidden}')
    oof,folds=base.build_oof(d,X,pc); roc=float(roc_auc_score(oof.y,oof.score)); pr=float(average_precision_score(oof.y,oof.score))
    w,r,search=choose(oof); wt,wm,alert_rate,green_cov,wworst,wper=w; rt,rm,red_n,rworst,rper=r
    gates={'roc_auc':roc>=CONTRACT['roc_auc_min'],'pr_auc':pr>=CONTRACT['pr_auc_min'],'red_precision':rm['precision']>=CONTRACT['red_precision_min'],'red_fpr':rm['FPR']<=CONTRACT['red_fpr_max'],'alert_recall':wm['recall']>=CONTRACT['alert_recall_min'],'alert_precision':wm['precision']>=CONTRACT['alert_precision_min'],'green_npv':wm['NPV']>=CONTRACT['green_npv_min'],'alert_rate':alert_rate<=CONTRACT['alert_rate_max'],'green_coverage':green_cov>=CONTRACT['green_coverage_min'],'worst_positive_fold_recall':wworst>=CONTRACT['worst_positive_fold_alert_recall_min'],'worst_fold_red_fpr':rworst<=CONTRACT['worst_fold_red_fpr_max']}
    fitted={name:base.fit_one(clone(factory),X,d.target) for name,factory in base.model_factories().items()}
    artifact={'version':VERSION,'models':fitted,'features':list(X.columns),'watch_threshold':wt,'red_threshold':rt,'min_precursor_red':2,'ood_profile':base.robust_ood_profile(X),'ood_max_fraction':.15,'expected_cities':sorted(set(panel.city.astype(str))),'development_end':str(base.DEV_END.date()),'target_definition':'next-week city POS value <80% current trailing-4-week mean','scope':'forecastable deterioration only; no claim of predicting signal-free exogenous shocks'}
    joblib.dump(artifact,MODEL)
    report={'version':VERSION,'rows':len(d),'positives':int(d.target.sum()),'positive_rate':float(d.target.mean()),'feature_count':len(X.columns),'ranking':{'ROC_AUC':roc,'PR_AUC':pr},'folds':folds,'search':search,'policy':{'watch_threshold':wt,'red_threshold':rt,'RED':rm,'RED_plus_AMBER':wm,'GREEN':{'NPV':wm['NPV'],'coverage':green_cov,'missed_declines':wm['FN']},'alert_rate':alert_rate,'alert_precision':wm['precision'],'red_alerts':red_n,'worst_positive_fold_alert_recall':wworst,'worst_fold_red_fpr':rworst},'watch_folds':wper,'red_folds':rper,'contract':CONTRACT,'gates':gates,'all_gates_passed':bool(all(gates.values())),'controls':{'no_city_identity':True,'no_target_prevalence_features':True,'no_absolute_levels':True,'compact_52_features':True,'multi_regime_oof':True,'selective_alert_coverage_constraint':True,'red_consensus_and_precursor_gate':True,'ood_fail_closed':True},'scientific_boundary':'No outcomes after 2025-06-29 used for fitting or threshold selection.'}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'development_summary.md').write_text('# Sales Sentinel v3.1 — Selective Triage\n\n'+f"- ROC-AUC **{roc:.2%}**, PR-AUC **{pr:.2%}**\n- RED precision **{rm['precision']:.2%}** ({rm['TP']} TP/{rm['FP']} FP), FPR **{rm['FPR']:.2%}**\n- RED+AMBER recall **{wm['recall']:.2%}**, precision **{wm['precision']:.2%}**, alert rate **{alert_rate:.2%}**\n- GREEN NPV **{wm['NPV']:.2%}**, coverage **{green_cov:.2%}**\n- Worst positive-fold recall **{wworst:.2%}**\n- All gates **{report['all_gates_passed']}**\n",encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
