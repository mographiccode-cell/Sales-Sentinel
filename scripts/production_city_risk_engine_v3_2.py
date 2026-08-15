from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import train_sama_city_risk_v3 as features

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models'/'sama_city_v3_2'/'city_risk_v3_2.joblib'
ENGINE_VERSION='SALES-SENTINEL-CITY-RISK-ENGINE-3.2'


def ood_fraction(row:pd.Series,profile:dict[str,dict[str,float]])->float:
    bad=0; total=0
    for c,p in profile.items():
        total+=1
        if c not in row.index or not np.isfinite(float(row[c])):
            bad+=1; continue
        x=float(row[c])
        if x<float(p['low']) or x>float(p['high']): bad+=1
    return float(bad/max(total,1))


def predict_latest(panel:pd.DataFrame,model_path:Path=MODEL)->dict[str,Any]:
    if not model_path.exists(): raise RuntimeError(f'v3.2 model artifact missing: {model_path}')
    artifact=joblib.load(model_path)
    src=panel.copy(); src['week_start']=pd.to_datetime(src['week_start'])
    required={'week_start','city','value_thousand_sar','transaction_count_thousand'}
    missing=required-set(src.columns)
    if missing: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'MISSING_SOURCE_COLUMNS:{sorted(missing)}'}
    if src.duplicated(['week_start','city']).any(): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'DUPLICATE_CITY_WEEK'}
    latest=pd.Timestamp(src.week_start.max()); latest_src=src[src.week_start.eq(latest)]
    expected=set(artifact.get('expected_cities',[])); actual=set(latest_src.city.astype(str))
    if not expected or actual!=expected:
        return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'CITY_CONTRACT_MISMATCH','expected_cities':sorted(expected),'actual_cities':sorted(actual)}
    if src.groupby('city').size().min()<104: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'INSUFFICIENT_HISTORY_LT_104_WEEKS'}
    if (latest_src[['value_thousand_sar','transaction_count_thousand']]<=0).any().any(): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'NONPOSITIVE_LATEST_SOURCE'}

    d,X,P,pc=features.featureize(src,require_target=False)
    missing_features=[c for c in artifact['features'] if c not in X.columns]
    if missing_features: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'FEATURE_SCHEMA_MISSING:{missing_features}'}
    mask=d.week_start.eq(latest)
    if int(mask.sum())!=len(expected): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'LATEST_FEATURE_ROWS_{int(mask.sum())}_NOT_{len(expected)}'}
    xx=X.loc[mask,artifact['features']].reset_index(drop=True); mm=d.loc[mask,['week_start','city']].reset_index(drop=True); pp=P.loc[mask].reset_index(drop=True); pcount=pc.loc[mask].reset_index(drop=True)
    if xx.isna().any().any(): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'LATEST_FEATURE_NULLS'}

    names=list(artifact['models']); model_scores={n:artifact['models'][n].predict_proba(xx)[:,1] for n in names}; matrix=np.column_stack([model_scores[n] for n in names]); score=matrix.mean(axis=1); agreement=(matrix>=.50).sum(axis=1)
    out=[]
    for i,row in mm.iterrows():
        ood=ood_fraction(xx.iloc[i],artifact['ood_profile']); precursor_names=[c for c in pp.columns if bool(pp.iloc[i][c])]
        red=(float(score[i])>=float(artifact['red_threshold']) and int(agreement[i])>=2 and int(pcount.iloc[i])>=int(artifact['min_precursor_red']))
        # Exact v3.2 parity: WATCH/AMBER is SCORE ONLY. Precursors gate RED but never create AMBER by themselves.
        watch=float(score[i])>=float(artifact['watch_threshold'])
        if ood>float(artifact['ood_max_fraction']): state='AMBER'; reason='OOD_ABSTAIN'
        elif red: state='RED'; reason='MODEL_CONSENSUS_AND_PRECURSORS'
        elif watch: state='AMBER'; reason='MODEL_EARLY_WARNING'
        else: state='GREEN'; reason='LOW_OBSERVED_RISK'
        out.append({'week_start':str(pd.Timestamp(row.week_start).date()),'city':row.city,'risk_score':float(score[i]),'model_scores':{n:float(model_scores[n][i]) for n in names},'model_agreement_ge_0_5':int(agreement[i]),'precursor_count':int(pcount.iloc[i]),'precursors':precursor_names,'ood_fraction':ood,'state':state,'reason':reason})
    return {'engine_version':ENGINE_VERSION,'model_version':artifact['version'],'status':'OK','latest_week':str(latest.date()),'predictions':out,'safety':{'future_label_used':False,'city_identity_feature_used':False,'target_prevalence_feature_used':False,'raw_absolute_level_feature_used':False,'ood_fail_closed_to_amber':True,'red_requires_model_agreement':True,'red_requires_observed_precursors':True,'watch_matches_training_policy':True,'surprise_shocks_claimed_predictable':False}}


def main():
    path=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'; panel=pd.read_csv(path,parse_dates=['week_start','week_end']); print(json.dumps(predict_latest(panel),indent=2))
if __name__=='__main__': main()
