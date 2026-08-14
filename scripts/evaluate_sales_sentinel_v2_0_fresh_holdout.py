from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import train_sales_sentinel_production_v2_0 as dev

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models'/'sales_sentinel_v2_0'/'sales_sentinel_market_risk_v2_0.joblib'
HISTORY=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2020_2026_extended.csv'
HOLDOUT=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2025_2026_holdout.csv'
FORECAST=ROOT/'data'/'sama_pos'/'sama_sector_walkforward_forecasts_2025_2026_v2_0.csv'
PARSE_AUDIT=ROOT/'reports'/'sama_recent_v2_0'/'parse_audit.json'
DEV_REPORT=ROOT/'reports'/'sales_sentinel_v2_0'/'development_report.json'
OUT=ROOT/'reports'/'sales_sentinel_v2_0'/'fresh_holdout_report.json'
SUMMARY=ROOT/'reports'/'sales_sentinel_v2_0'/'fresh_holdout_summary.md'


def wilson(k,n,z=1.96):
    if n<=0: return [None,None]
    p=k/n; den=1+z*z/n; center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.0,center-half),min(1.0,center+half)]


def cal(y,p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6); y=np.asarray(y,int)
    bins=np.linspace(0,1,11); ece=0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(p>=lo)&((p<hi) if hi<1 else (p<=hi))
        if m.any(): ece += m.mean()*abs(p[m].mean()-y[m].mean())
    return {'Brier':float(brier_score_loss(y,p)),'ECE_10bin':float(ece)}


def triage_details(y,p,watch,red):
    y=np.asarray(y,int); p=np.asarray(p,float)
    state=np.where(p>=red,'RED',np.where(p>=watch,'AMBER','GREEN'))
    redm=state=='RED'; amber=state=='AMBER'; green=state=='GREEN'; alert=~green
    positives=int((y==1).sum()); negatives=int((y==0).sum())
    red_tp=int(((y==1)&redm).sum()); red_fp=int(((y==0)&redm).sum())
    alert_tp=int(((y==1)&alert).sum()); alert_fp=int(((y==0)&alert).sum())
    green_tn=int(((y==0)&green).sum()); green_fn=int(((y==1)&green).sum())
    red_precision=red_tp/max(red_tp+red_fp,1); alert_recall=alert_tp/max(positives,1); green_npv=green_tn/max(green_tn+green_fn,1)
    out={
        'rows':int(len(y)),'declines':positives,'positive_rate':positives/max(len(y),1),
        'RED':{'rows':int(redm.sum()),'coverage':float(redm.mean()),'TP':red_tp,'FP':red_fp,'precision':red_precision,'precision_wilson95':wilson(red_tp,red_tp+red_fp),'recall_contribution':red_tp/max(positives,1),'false_positive_rate':red_fp/max(negatives,1)},
        'AMBER':{'rows':int(amber.sum()),'coverage':float(amber.mean()),'declines':int(((y==1)&amber).sum()),'positive_rate':float(((y==1)&amber).sum()/max(amber.sum(),1))},
        'GREEN':{'rows':int(green.sum()),'coverage':float(green.mean()),'TN':green_tn,'FN':green_fn,'NPV':green_npv,'NPV_wilson95':wilson(green_tn,green_tn+green_fn),'miss_rate_of_all_declines':green_fn/max(positives,1)},
        'RED_plus_AMBER':{'rows':int(alert.sum()),'coverage':float(alert.mean()),'TP':alert_tp,'FP':alert_fp,'precision':alert_tp/max(alert_tp+alert_fp,1),'recall':alert_recall,'recall_wilson95':wilson(alert_tp,positives)},
    }
    return out,state


def main():
    artifact=joblib.load(MODEL)
    parse=json.load(open(PARSE_AUDIT,encoding='utf-8'))
    development=json.load(open(DEV_REPORT,encoding='utf-8'))
    if not parse.get('all_checks_passed'): raise RuntimeError('Fresh holdout quality gate did not pass')
    if not development['separation']['fresh_holdout_not_used']: raise RuntimeError('Development report indicates holdout contamination')
    if artifact['version']!='SALES-SENTINEL-MARKET-RISK-2.0-FROZEN': raise RuntimeError('Unexpected model version')
    if artifact['selected_score']!='ForecastRaw':
        raise RuntimeError(f"This sealed evaluator expects the frozen selected score ForecastRaw; got {artifact['selected_score']}")

    hist=pd.read_csv(HISTORY,parse_dates=['week_start','week_end']).sort_values(['sector','week_start'])
    hold=pd.read_csv(HOLDOUT,parse_dates=['week_start','week_end'])
    fc=pd.read_csv(FORECAST,parse_dates=['origin_week_start','forecast_h1_week_start'])
    hist['baseline4']=hist.groupby('sector').value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
    base=hist[['week_start','sector','baseline4']].rename(columns={'week_start':'origin_week_start'})
    d=fc.merge(base,on=['origin_week_start','sector'],how='inner',validate='many_to_one')
    d=d[d.origin_week_start>=hold.week_start.min()].dropna(subset=['baseline4','predicted_value_h1','actual_value_h1']).copy()
    # Only evaluate predictions whose target week is inside the fresh PDF holdout.
    valid_targets=set(zip(hold.week_start.dt.normalize(),hold.sector))
    d=d[[ (pd.Timestamp(w).normalize(),s) in valid_targets for w,s in zip(d.forecast_h1_week_start,d.sector) ]].copy()
    if len(d)<400: raise RuntimeError(f'Too few fresh holdout predictions: {len(d)}')

    d['target']=(d.actual_value_h1 < (1-dev.DECLINE)*d.baseline4).astype(int)
    d['pred_value_ratio4']=d.predicted_value_h1/d.baseline4
    raw=dev.sigmoid(((1-dev.DECLINE)-d.pred_value_ratio4.to_numpy())/.055)
    # Frozen Platt calibrator learned on 2024 OOF only.
    p=artifact['calibrator'].predict_proba(pd.DataFrame({'ForecastRaw':raw}))[:,1]
    d['risk_probability']=p
    watch=float(artifact['watch_threshold']); red=float(artifact['red_threshold'])
    tri,state=triage_details(d.target,p,watch,red); d['state']=state
    ranking={'ROC_AUC':float(roc_auc_score(d.target,p)),'PR_AUC':float(average_precision_score(d.target,p))}
    calibration=cal(d.target,p)

    gates={
        'red_precision':tri['RED']['precision']>=artifact['acceptance_contract']['red_precision_min'],
        'red_false_positive_rate':tri['RED']['false_positive_rate']<=artifact['acceptance_contract']['red_false_positive_rate_max'],
        'alert_recall':tri['RED_plus_AMBER']['recall']>=artifact['acceptance_contract']['alert_recall_min'],
        'green_npv':tri['GREEN']['NPV']>=artifact['acceptance_contract']['green_npv_min'],
        'roc_auc':ranking['ROC_AUC']>=artifact['acceptance_contract']['roc_auc_min'],
        'pr_auc':ranking['PR_AUC']>=artifact['acceptance_contract']['pr_auc_min'],
    }

    # Stability diagnostics; never used to alter thresholds.
    d['half']=np.where(d.origin_week_start < d.origin_week_start.min()+(d.origin_week_start.max()-d.origin_week_start.min())/2,'FIRST_HALF','SECOND_HALF')
    temporal={}
    for name,q in d.groupby('half'):
        if q.target.nunique()<2: continue
        t,_=triage_details(q.target,q.risk_probability,watch,red)
        temporal[name]={'rows':int(len(q)),'triage':t,'ROC_AUC':float(roc_auc_score(q.target,q.risk_probability)),'PR_AUC':float(average_precision_score(q.target,q.risk_probability))}
    sector_diag={}
    for sector,q in d.groupby('sector'):
        if len(q)<20: continue
        tt,_=triage_details(q.target,q.risk_probability,watch,red)
        sector_diag[sector]={'rows':int(len(q)),'declines':int(q.target.sum()),'triage':tt}

    report={
        'version':'SALES-SENTINEL-V2.0-FRESH-HOLDOUT-EVALUATION',
        'model_version':artifact['version'],'model_was_frozen_before_holdout':True,
        'holdout_source':'Official Saudi Central Bank (SAMA) weekly POS PDFs acquired after model-development period',
        'holdout_rows':int(len(d)),'holdout_origins':int(d.origin_week_start.nunique()),'sectors':int(d.sector.nunique()),
        'origin_start':str(d.origin_week_start.min().date()),'origin_end':str(d.origin_week_start.max().date()),
        'target_week_end':str(d.forecast_h1_week_start.max().date()),
        'fixed_thresholds':{'watch':watch,'red':red},
        'triage':tri,'ranking':ranking,'calibration':calibration,
        'acceptance_contract':artifact['acceptance_contract'],'gates':gates,'all_gates_passed':bool(all(gates.values())),
        'temporal_stability':temporal,'sector_diagnostics':sector_diag,
        'anti_leakage_attestation':{
            'fresh_holdout_not_used_for_model_selection':True,'fresh_holdout_not_used_for_calibration':True,'fresh_holdout_not_used_for_threshold_selection':True,
            'thresholds_loaded_from_pre_holdout_frozen_artifact':True,'forecaster_hyperparameters_frozen_from_v1_7':True,'walk_forward_forecasts_use_only_information_available_at_each_origin':True,
            'no_shuffle':True,
        },
    }
    OUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    SUMMARY.write_text(f'''# Sales Sentinel v2.0 — Fresh Unseen SAMA Holdout\n\n- Frozen before holdout: **Yes**\n- Holdout predictions: **{len(d):,}**\n- Sectors: **{d.sector.nunique()}**\n- Origin period: **{d.origin_week_start.min().date()} → {d.origin_week_start.max().date()}**\n- Decline rate: **{d.target.mean():.2%}**\n- RED precision: **{tri['RED']['precision']:.2%}** ({tri['RED']['TP']} true / {tri['RED']['FP']} false alerts)\n- RED false-positive rate: **{tri['RED']['false_positive_rate']:.2%}**\n- RED+AMBER recall: **{tri['RED_plus_AMBER']['recall']:.2%}**\n- Missed declines in GREEN: **{tri['GREEN']['FN']} / {tri['declines']}**\n- GREEN NPV: **{tri['GREEN']['NPV']:.2%}**\n- PR-AUC: **{ranking['PR_AUC']:.2%}**\n- ROC-AUC: **{ranking['ROC_AUC']:.2%}**\n- Brier: **{calibration['Brier']:.4f}**\n- All frozen production gates passed: **{report['all_gates_passed']}**\n''',encoding='utf-8')
    print(json.dumps({
        'rows':len(d),'period':[str(d.origin_week_start.min().date()),str(d.origin_week_start.max().date())],
        'triage':tri,'ranking':ranking,'calibration':calibration,'gates':gates,'all_gates':report['all_gates_passed'],
    },indent=2))

if __name__=='__main__': main()
