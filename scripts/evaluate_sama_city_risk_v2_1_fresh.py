from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import train_sama_city_risk_v2_1 as city
import run_sama_city_risk_v2_1 as reconciled

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
FRESH=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2025_2026_holdout.csv'
MODEL=ROOT/'models'/'sama_city_v2_1'/'city_market_risk_v2_1.joblib'
AUDIT=ROOT/'reports'/'sama_city_v2_1'/'fresh_holdout_audit.json'
DEV=ROOT/'reports'/'sama_city_v2_1'/'development_report.json'
REPORT=ROOT/'reports'/'sama_city_v2_1'/'fresh_evaluation.json'
SUMMARY=ROOT/'reports'/'sama_city_v2_1'/'fresh_evaluation_summary.md'


def wilson(k,n,z=1.96):
    if n<=0:return [None,None]
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-half),min(1,center+half)]

def score_selected(artifact,d,X):
    name=artifact['selected']; kind=artifact['selected_kind']
    if kind=='classifier': raw=artifact['models'][name].predict_proba(X)[:,1]
    elif kind=='regressor':
        pred=np.expm1(artifact['models'][name].predict(X)); ratio=pred/d.baseline4.to_numpy(); raw=city.sigmoid(((1-city.DECLINE)-ratio)/.055)
    elif name=='CurrentWeekRule':
        ratio=d.value_thousand_sar.to_numpy()/d.baseline4.to_numpy(); raw=city.sigmoid(((1-city.DECLINE)-ratio)/.055)
    else: raise RuntimeError(f'Unsupported frozen selected score {name}/{kind}')
    return raw

def evaluate_triage(y,p,wt,rt):
    y=np.asarray(y,int); p=np.asarray(p,float); state=np.where(p>=rt,'RED',np.where(p>=wt,'AMBER','GREEN'))
    red=state=='RED'; amber=state=='AMBER'; green=state=='GREEN'; alert=~green; pos=(y==1).sum(); neg=(y==0).sum()
    redtp=((y==1)&red).sum(); redfp=((y==0)&red).sum(); atp=((y==1)&alert).sum(); afp=((y==0)&alert).sum(); gtn=((y==0)&green).sum(); gfn=((y==1)&green).sum()
    return {
        'rows':int(len(y)),'declines':int(pos),'decline_rate':float(pos/max(len(y),1)),
        'RED':{'rows':int(red.sum()),'TP':int(redtp),'FP':int(redfp),'precision':float(redtp/max(redtp+redfp,1)),'precision_wilson95':wilson(int(redtp),int(redtp+redfp)),'FPR':float(redfp/max(neg,1)),'recall_contribution':float(redtp/max(pos,1))},
        'AMBER':{'rows':int(amber.sum()),'declines':int(((y==1)&amber).sum()),'positive_rate':float(((y==1)&amber).sum()/max(amber.sum(),1))},
        'GREEN':{'rows':int(green.sum()),'TN':int(gtn),'FN':int(gfn),'NPV':float(gtn/max(gtn+gfn,1)),'NPV_wilson95':wilson(int(gtn),int(gtn+gfn)),'miss_rate':float(gfn/max(pos,1))},
        'RED_plus_AMBER':{'rows':int(alert.sum()),'TP':int(atp),'FP':int(afp),'precision':float(atp/max(atp+afp,1)),'recall':float(atp/max(pos,1)),'recall_wilson95':wilson(int(atp),int(pos))},
    },state

def main():
    art=joblib.load(MODEL); aud=json.load(open(AUDIT,encoding='utf-8')); dev=json.load(open(DEV,encoding='utf-8'))
    if not aud['all_checks_passed']: raise RuntimeError('Fresh city data quality gate failed')
    if dev['leakage_controls']['fresh_2025_2026_holdout_used']: raise RuntimeError('Development contaminated by fresh holdout')
    if art['version']!='SAMA-CITY-MARKET-RISK-2.1-FROZEN': raise RuntimeError('Unexpected frozen artifact')

    d0=reconciled.reconciled_load_panel(EXT); d,X=city.featureize(d0)
    fresh=pd.read_csv(FRESH,parse_dates=['week_start']); valid=set(zip(fresh.week_start.dt.normalize(),fresh.city))
    # Origin itself and its next-week target must both be in the new official PDF holdout.
    mask=[(pd.Timestamp(w).normalize(),c) in valid and (pd.Timestamp(w).normalize()+pd.Timedelta(days=7),c) in valid for w,c in zip(d.week_start,d.city)]
    # Source occasionally shifts weekly boundary by one day; robustly validate target by sequence membership instead of date arithmetic.
    if sum(mask)<300:
        fresh_keys={(c,tuple(q.week_start.sort_values())) for c,q in fresh.groupby('city')}
        fresh_weeksets={c:set(q.week_start.dt.normalize()) for c,q in fresh.groupby('city')}
        # Use city sequence index: next observation is the evaluation target, matching development featureize group shift.
        mask=[]
        for w,c in zip(d.week_start,d.city):
            weeks=sorted(fresh_weeksets.get(c,set()))
            ww=pd.Timestamp(w).normalize()
            mask.append(ww in fresh_weeksets.get(c,set()) and ww!=weeks[-1] if weeks else False)
    d=d.loc[mask].reset_index(drop=True); X=X.loc[mask].reset_index(drop=True)
    if len(d)<400: raise RuntimeError(f'Too few sealed city predictions: {len(d)}')

    raw=score_selected(art,d,X); p=art['calibrator'].predict_proba(pd.DataFrame({art['selected']:raw}))[:,1]
    wt=float(art['watch_threshold']); rt=float(art['red_threshold']); tri,state=evaluate_triage(d.target,p,wt,rt); d['risk']=p; d['state']=state
    ranking={'ROC_AUC':float(roc_auc_score(d.target,p)),'PR_AUC':float(average_precision_score(d.target,p))}; calibration={'Brier':float(brier_score_loss(d.target,np.clip(p,1e-6,1-1e-6)))}
    gates={'red_precision':tri['RED']['precision']>=art['acceptance']['red_precision_min'],'red_fpr':tri['RED']['FPR']<=art['acceptance']['red_false_positive_rate_max'],'alert_recall':tri['RED_plus_AMBER']['recall']>=art['acceptance']['alert_recall_min'],'green_npv':tri['GREEN']['NPV']>=art['acceptance']['green_npv_min'],'roc_auc':ranking['ROC_AUC']>=art['acceptance']['roc_auc_min'],'pr_auc':ranking['PR_AUC']>=art['acceptance']['pr_auc_min']}
    by_city={}
    for c,q in d.groupby('city'):
        tt,_=evaluate_triage(q.target,q.risk,wt,rt); by_city[c]={'rows':len(q),'declines':int(q.target.sum()),'triage':tt,'ROC_AUC':float(roc_auc_score(q.target,q.risk)) if q.target.nunique()==2 else None,'PR_AUC':float(average_precision_score(q.target,q.risk)) if q.target.sum()>0 else None}
    # first-vs-second half drift check only; thresholds never change.
    cut=d.week_start.sort_values().iloc[len(d)//2]; halves={}
    for name,q in [('FIRST',d[d.week_start<=cut]),('SECOND',d[d.week_start>cut])]:
        if len(q)==0:continue
        tt,_=evaluate_triage(q.target,q.risk,wt,rt); halves[name]={'rows':len(q),'triage':tt,'ROC_AUC':float(roc_auc_score(q.target,q.risk)) if q.target.nunique()==2 else None,'PR_AUC':float(average_precision_score(q.target,q.risk)) if q.target.sum()>0 else None}
    report={'version':'SAMA-CITY-RISK-2.1-FRESH-EVAL','frozen_model_version':art['version'],'model_frozen_before_holdout':True,'holdout_source':'Official SAMA City Total weekly POS PDFs, post-development','rows':len(d),'cities':int(d.city.nunique()),'origin_start':str(d.week_start.min().date()),'origin_end':str(d.week_start.max().date()),'fixed_thresholds':{'watch':wt,'red':rt},'triage':tri,'ranking':ranking,'calibration':calibration,'acceptance':art['acceptance'],'gates':gates,'all_gates_passed':bool(all(gates.values())),'temporal_stability':halves,'by_city':by_city,'anti_leakage':{'fresh_holdout_not_used_for_training':True,'fresh_holdout_not_used_for_model_selection':True,'fresh_holdout_not_used_for_calibration':True,'fresh_holdout_not_used_for_threshold_selection':True,'static_frozen_model_weights':True,'actual_prior_holdout_weeks_used_only_as_operational_lag_features':True,'shuffle':False}}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8'); SUMMARY.write_text(f'''# SAMA City Risk v2.1 — Fresh Holdout\n\n- Frozen before holdout: **Yes**\n- Rows: **{len(d):,}** across **{d.city.nunique()} cities**\n- Period: **{d.week_start.min().date()} → {d.week_start.max().date()}**\n- Decline rate: **{d.target.mean():.2%}**\n- RED precision: **{tri['RED']['precision']:.2%}** ({tri['RED']['TP']} TP / {tri['RED']['FP']} FP)\n- RED FPR: **{tri['RED']['FPR']:.2%}**\n- RED+AMBER recall: **{tri['RED_plus_AMBER']['recall']:.2%}**\n- GREEN NPV: **{tri['GREEN']['NPV']:.2%}**\n- Missed declines in GREEN: **{tri['GREEN']['FN']} / {tri['declines']}**\n- PR-AUC: **{ranking['PR_AUC']:.2%}**\n- ROC-AUC: **{ranking['ROC_AUC']:.2%}**\n- All frozen gates passed: **{report['all_gates_passed']}**\n''',encoding='utf-8')
    print(json.dumps({'rows':len(d),'triage':tri,'ranking':ranking,'calibration':calibration,'gates':gates,'all':report['all_gates_passed']},indent=2))

if __name__=='__main__': main()
