from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import train_sama_city_risk_v3 as features

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models'/'sama_city_v3_3'/'city_risk_v3_3.joblib'
ENGINE_VERSION='SALES-SENTINEL-CITY-RISK-ENGINE-3.3'

def ood_fraction(row,profile):
    bad=0; total=0
    for c,p in profile.items():
        total+=1
        if c not in row.index or not np.isfinite(float(row[c])): bad+=1; continue
        x=float(row[c]); bad+=int(x<float(p['low']) or x>float(p['high']))
    return float(bad/max(total,1))

def predict_latest(panel:pd.DataFrame,model_path:Path=MODEL)->dict[str,Any]:
    if not model_path.exists(): raise RuntimeError(f'v3.3 model missing: {model_path}')
    a=joblib.load(model_path); src=panel.copy(); src['week_start']=pd.to_datetime(src.week_start)
    req={'week_start','city','value_thousand_sar','transaction_count_thousand'}; miss=req-set(src.columns)
    if miss: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'MISSING_SOURCE_COLUMNS:{sorted(miss)}'}
    if src.duplicated(['week_start','city']).any(): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'DUPLICATE_CITY_WEEK'}
    latest=pd.Timestamp(src.week_start.max()); q=src[src.week_start.eq(latest)]; expected=set(a['expected_cities']); actual=set(q.city.astype(str))
    if actual!=expected: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'CITY_CONTRACT_MISMATCH','expected':sorted(expected),'actual':sorted(actual)}
    if src.groupby('city').size().min()<104: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'INSUFFICIENT_HISTORY_LT_104_WEEKS'}
    if (q[['value_thousand_sar','transaction_count_thousand']]<=0).any().any(): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'NONPOSITIVE_LATEST_SOURCE'}
    d,X,P,pc=features.featureize(src,require_target=False); missing=[c for c in a['features'] if c not in X.columns]
    if missing: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'FEATURE_SCHEMA_MISSING:{missing}'}
    mask=d.week_start.eq(latest)
    if int(mask.sum())!=len(expected): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'LATEST_FEATURE_ROWS_{int(mask.sum())}'}
    xx=X.loc[mask,a['features']].reset_index(drop=True); mm=d.loc[mask,['week_start','city']].reset_index(drop=True); pp=P.loc[mask].reset_index(drop=True); pcount=pc.loc[mask].reset_index(drop=True)
    if xx.isna().any().any(): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'LATEST_FEATURE_NULLS'}
    names=list(a['models']); scores={n:a['models'][n].predict_proba(xx)[:,1] for n in names}; mat=np.column_stack([scores[n] for n in names]); risk=mat.mean(axis=1); agreement=(mat>=.5).sum(axis=1)
    out=[]
    for i,row in mm.iterrows():
        ood=ood_fraction(xx.iloc[i],a['ood_profile']); pci=int(pcount.iloc[i]); sc=float(risk[i]); precursor_names=[c for c in pp.columns if bool(pp.iloc[i][c])]
        red=sc>=float(a['red_threshold']) and int(agreement[i])>=2 and pci>=int(a['min_precursor_red'])
        primary=sc>=float(a['watch_threshold']); fallback=pci>=int(a['high_precursor_count']) and sc>=float(a['high_precursor_fallback_threshold'])
        if ood>float(a['ood_max_fraction']): state='AMBER'; reason='OOD_ABSTAIN'
        elif red: state='RED'; reason='MODEL_CONSENSUS_AND_PRECURSORS'
        elif primary: state='AMBER'; reason='MODEL_EARLY_WARNING'
        elif fallback: state='AMBER'; reason='HIGH_PRECURSOR_FALLBACK'
        else: state='GREEN'; reason='LOW_OBSERVED_RISK'
        out.append({'week_start':str(pd.Timestamp(row.week_start).date()),'city':row.city,'risk_score':sc,'model_scores':{n:float(scores[n][i]) for n in names},'model_agreement_ge_0_5':int(agreement[i]),'precursor_count':pci,'precursors':precursor_names,'ood_fraction':ood,'state':state,'reason':reason})
    return {'engine_version':ENGINE_VERSION,'model_version':a['version'],'status':'OK','latest_week':str(latest.date()),'predictions':out,'safety':{'future_label_used':False,'city_identity_feature_used':False,'target_prevalence_feature_used':False,'absolute_level_feature_used':False,'dual_channel_policy_frozen':True,'ood_fail_closed_to_amber':True,'surprise_shocks_claimed_predictable':False}}

def main():
    p=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'; d=pd.read_csv(p,parse_dates=['week_start','week_end']); print(json.dumps(predict_latest(d),indent=2))
if __name__=='__main__': main()
