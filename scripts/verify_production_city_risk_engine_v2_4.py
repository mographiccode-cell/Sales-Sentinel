from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import production_city_risk_engine_v2_4 as prod
import run_sama_city_risk_v2_1 as source
import train_sama_city_risk_v2_2 as train

ROOT=Path(__file__).resolve().parents[1]
HIST=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
EXT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
OUT=ROOT/'reports'/'sama_city_v2_4'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'production_engine_verification.json'


def main():
    base=joblib.load(prod.BASE_MODEL)
    hist=pd.read_csv(HIST,parse_dates=['week_start','week_end'])
    ext=pd.read_csv(EXT,parse_dates=['week_start','week_end'])

    td,tx=train.featureize(source.reconciled_load_panel(HIST))
    idf,ix=prod.build_inference_features(hist)
    left=pd.concat([td[['week_start','city']],tx[base['features']]],axis=1)
    right=pd.concat([idf[['week_start','city']],ix[base['features']]],axis=1)
    m=left.merge(right,on=['week_start','city'],suffixes=('_training','_production'),validate='one_to_one')
    max_parity=0.0
    for c in base['features']:
        a=m[f'{c}_training'].to_numpy(float); b=m[f'{c}_production'].to_numpy(float)
        diff=np.nanmax(np.abs(a-b)) if len(a) else 0.0
        max_parity=max(max_parity,float(diff))

    origin=pd.Timestamp('2025-05-04')
    fd,fx=prod.build_inference_features(hist)
    td2,tx2=prod.build_inference_features(hist[hist.week_start<=origin])
    future_diff=0.0
    for city in sorted(td2.city.unique()):
        fm=fd.week_start.eq(origin)&fd.city.eq(city); tm=td2.week_start.eq(origin)&td2.city.eq(city)
        a=fx.loc[fm,base['features']].to_numpy(float); b=tx2.loc[tm,base['features']].to_numpy(float)
        if a.size and b.size: future_diff=max(future_diff,float(np.nanmax(np.abs(a-b))))

    latest_result=prod.predict_latest(ext)
    latest=ext.week_start.max(); victim=sorted(ext.loc[ext.week_start.eq(latest),'city'].unique())[0]
    broken=ext[~(ext.week_start.eq(latest)&ext.city.eq(victim))].copy()
    broken_result=prod.predict_latest(broken)

    checks={
        'training_vs_production_feature_max_abs_diff_le_1e12':max_parity<=1e-12,
        'truncated_future_feature_max_abs_diff_le_1e12':future_diff<=1e-12,
        'latest_unknown_target_prediction_status_ok':latest_result.get('status')=='OK',
        'latest_returns_exactly_11_city_predictions':len(latest_result.get('predictions',[]))==11,
        'latest_prediction_declares_future_label_unused':latest_result.get('safety',{}).get('future_label_used') is False,
        'missing_city_fails_closed':broken_result.get('status')=='NO_DECISION' and broken_result.get('state')=='AMBER',
        'missing_city_reason_is_source_qc':broken_result.get('reason')=='SOURCE_QC_FAILED',
    }
    report={
        'version':'SALES-SENTINEL-CITY-RISK-ENGINE-2.4-VERIFICATION',
        'base_model':base['version'],
        'feature_count':len(base['features']),
        'historical_parity_rows':len(m),
        'training_vs_production_max_abs_feature_difference':max_parity,
        'future_truncation_max_abs_feature_difference':future_diff,
        'latest_inference':{
            'status':latest_result.get('status'),
            'week':str(latest),
            'prediction_count':len(latest_result.get('predictions',[])),
            'state_counts':pd.Series([x['state'] for x in latest_result.get('predictions',[])]).value_counts().to_dict(),
        },
        'fail_closed_test':{'removed_city':victim,'status':broken_result.get('status'),'state':broken_result.get('state'),'reason':broken_result.get('reason')},
        'checks':checks,
        'all_checks_passed':bool(all(checks.values())),
    }
    REPORT.write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))
    if not report['all_checks_passed']:
        raise RuntimeError('Production city risk engine v2.4 verification failed')

if __name__=='__main__':main()
