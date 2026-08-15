from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import train_sama_city_risk_v3 as features
import train_sama_city_risk_v3_4_1 as helpers

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models'/'sama_city_v3_4_2'/'city_risk_v3_4_2.joblib'
ENGINE_VERSION='SALES-SENTINEL-CITY-RISK-ENGINE-3.4.2'
EXPECTED_VERSION='SAMA-CITY-RISK-3.4.2-DOWNSIDE-RATIO'


def ood_fraction(row,profile):
    bad=0; total=0
    for c,p in profile.items():
        total+=1
        if c not in row.index or not np.isfinite(float(row[c])):
            bad+=1; continue
        x=float(row[c]); bad+=int(x<float(p['low']) or x>float(p['high']))
    return float(bad/max(total,1))


def predict_latest(panel:pd.DataFrame,model_path:Path=MODEL)->dict[str,Any]:
    if not model_path.exists(): raise RuntimeError(f'v3.4.2 model missing: {model_path}')
    a=joblib.load(model_path)
    if a.get('version')!=EXPECTED_VERSION: raise RuntimeError(f'wrong artifact {a.get("version")}')
    src=panel.copy(); src['week_start']=pd.to_datetime(src.week_start)
    req={'week_start','city','value_thousand_sar','transaction_count_thousand'}; miss=req-set(src.columns)
    if miss: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'MISSING_SOURCE_COLUMNS:{sorted(miss)}'}
    if src.duplicated(['week_start','city']).any(): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'DUPLICATE_CITY_WEEK'}
    latest=pd.Timestamp(src.week_start.max()); latest_src=src[src.week_start.eq(latest)]; expected=set(a['expected_cities']); actual=set(latest_src.city.astype(str))
    if actual!=expected: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'CITY_CONTRACT_MISMATCH','expected':sorted(expected),'actual':sorted(actual)}
    if src.groupby('city').size().min()<104: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'INSUFFICIENT_HISTORY_LT_104_WEEKS'}
    if (latest_src[['value_thousand_sar','transaction_count_thousand']]<=0).any().any(): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'NONPOSITIVE_LATEST_SOURCE'}

    d,X,P,pc=features.featureize(src,require_target=False)
    missing=[c for c in a['features'] if c not in X.columns]
    ratio_missing=[c for c in a['ratio_features'] if c not in X.columns]
    if missing or ratio_missing: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'FEATURE_SCHEMA_MISSING:{sorted(set(missing+ratio_missing))}'}
    mask=d.week_start.eq(latest)
    if int(mask.sum())!=len(expected): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'LATEST_FEATURE_ROWS_{int(mask.sum())}'}
    xx=X.loc[mask,a['features']].reset_index(drop=True); xr=X.loc[mask,a['ratio_features']].reset_index(drop=True); mm=d.loc[mask,['week_start','city']].reset_index(drop=True); pp=P.loc[mask].reset_index(drop=True); pcount=pc.loc[mask].reset_index(drop=True)
    if xx.isna().any().any() or xr.isna().any().any(): return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'LATEST_FEATURE_NULLS'}

    names=list(a['models']); ms={n:a['models'][n].predict_proba(xx)[:,1] for n in names}; mat=np.column_stack([ms[n] for n in names]); risk=mat.mean(axis=1); agreement=(mat>=.5).sum(axis=1)
    rnames=list(a['ratio_models']); rpred={n:np.exp(a['ratio_models'][n].predict(xr)) for n in rnames}; ratio=np.column_stack([rpred[n] for n in rnames]).mean(axis=1)
    evidence=helpers.evidence_count(xr)

    out=[]
    for i,row in mm.iterrows():
        sc=float(risk[i]); pci=int(pcount.iloc[i]); ev=int(evidence[i]); pr=float(ratio[i]); ood=ood_fraction(xx.iloc[i],a['ood_profile']); precursor_names=[c for c in pp.columns if bool(pp.iloc[i][c])]
        red=sc>=float(a['red_threshold']) and int(agreement[i])>=2 and pci>=int(a['min_precursor_red'])
        base_watch=sc>=float(a['watch_threshold'])
        high_precursor=pci>=int(a['high_precursor_count']) and sc>=float(a['high_precursor_fallback_threshold'])
        downside=pr<=float(a['ratio_threshold']) and ev>=int(a['ratio_evidence_min'])
        if ood>float(a['ood_max_fraction']): state='AMBER'; reason='OOD_ABSTAIN'
        elif red: state='RED'; reason='MODEL_CONSENSUS_AND_PRECURSORS'
        elif base_watch: state='AMBER'; reason='MODEL_EARLY_WARNING'
        elif high_precursor: state='AMBER'; reason='HIGH_PRECURSOR_FALLBACK'
        elif downside: state='AMBER'; reason='DOWNSIDE_RATIO_FORECAST'
        else: state='GREEN'; reason='LOW_OBSERVED_RISK'
        out.append({'week_start':str(pd.Timestamp(row.week_start).date()),'city':row.city,'risk_score':sc,'model_scores':{n:float(ms[n][i]) for n in names},'model_agreement_ge_0_5':int(agreement[i]),'precursor_count':pci,'precursors':precursor_names,'predicted_next_week_ratio':pr,'ratio_model_predictions':{n:float(rpred[n][i]) for n in rnames},'trend_evidence_count':ev,'ood_fraction':ood,'state':state,'reason':reason})
    return {'engine_version':ENGINE_VERSION,'model_version':a['version'],'status':'OK','latest_week':str(latest.date()),'predictions':out,'safety':{'future_label_used':False,'city_identity_feature_used':False,'target_prevalence_feature_used':False,'absolute_level_feature_used':False,'ratio_target_used_only_during_training':True,'ood_fail_closed_to_amber':True,'red_policy_inherited_unchanged_from_v3_3':True,'surprise_shocks_claimed_predictable':False}}


def main():
    p=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'; d=pd.read_csv(p,parse_dates=['week_start','week_end']); print(json.dumps(predict_latest(d),indent=2))
if __name__=='__main__':main()
