from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

import build_prior_shift_policy_v2_4 as prior

ROOT=Path(__file__).resolve().parents[1]
BASE_MODEL=ROOT/'models'/'sama_city_v2_2'/'city_market_risk_v2_2.joblib'
POLICY_MODEL=ROOT/'models'/'sama_city_v2_4'/'prior_shift_policy_v2_4.joblib'
ENGINE_VERSION='SALES-SENTINEL-CITY-RISK-ENGINE-2.4.2'


def safe_ratio(a,b):
    a=pd.Series(a,index=getattr(a,'index',None),dtype=float)
    b=pd.Series(b,index=a.index,dtype=float).replace(0,np.nan)
    return a/b


def build_inference_features(panel:pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Build the v2.2 frozen feature contract WITHOUT requiring the unknown next-week label.

    Historical targets are computed only to create shifted prevalence features. At origin t, target(t)
    is unknown and never used; the newest target contributing to a feature is target(t-1), whose
    outcome is the already-observed week t value.
    """
    required={'week_start','city','value_thousand_sar','transaction_count_thousand'}
    missing=required-set(panel.columns)
    if missing: raise ValueError(f'Missing required source columns: {sorted(missing)}')
    d=panel.copy(); d['week_start']=pd.to_datetime(d.week_start); d=d.sort_values(['city','week_start']).reset_index(drop=True)
    if d.duplicated(['week_start','city']).any(): raise ValueError('Duplicate week/city rows')
    g=d.groupby('city',sort=False)

    d['baseline4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
    d['actual_next_value']=g.value_thousand_sar.shift(-1)
    d['future_ratio']=safe_ratio(d.actual_next_value,d.baseline4)
    d['target_float']=np.where(d.future_ratio.notna(),(d.future_ratio < .80).astype(float),np.nan)
    d['target']=d.target_float.fillna(0).astype(int)

    F=pd.DataFrame(index=d.index)
    for col,pre in [('value_thousand_sar','value'),('transaction_count_thousand','count')]:
        s=d[col].astype(float)
        for w in (4,8,13,26,52):
            mean=g[col].transform(lambda x,w=w:x.rolling(w,min_periods=w).mean())
            std=g[col].transform(lambda x,w=w:x.rolling(w,min_periods=w).std())
            F[f'{pre}_ratio_mean_{w}']=safe_ratio(s,mean)
            F[f'{pre}_cv_{w}']=safe_ratio(std,mean.abs())
        for lag in (1,2,4,8,13,26,52): F[f'{pre}_change_{lag}']=g[col].pct_change(lag)
        F[f'{pre}_yoy_log_ratio']=np.log(s/g[col].shift(52).replace(0,np.nan))

    national=d.groupby('week_start',as_index=False).agg(nvalue=('value_thousand_sar','sum'),ncount=('transaction_count_thousand','sum'))
    d=d.merge(national,on='week_start',how='left',validate='many_to_one')
    d['value_share']=safe_ratio(d.value_thousand_sar,d.nvalue); d['count_share']=safe_ratio(d.transaction_count_thousand,d.ncount)
    gs=d.groupby('city',sort=False)
    for col in ('value_share','count_share'):
        for w in (4,13,26,52):
            mean=gs[col].transform(lambda s,w=w:s.rolling(w,min_periods=w).mean())
            F[f'{col}_ratio_{w}']=safe_ratio(d[col],mean)
        for lag in (1,4,13,52): F[f'{col}_change_{lag}']=gs[col].pct_change(lag)

    n=national.sort_values('week_start').copy()
    for col,pre in [('nvalue','nvalue'),('ncount','ncount')]:
        for w in (4,8,13,26,52):
            mean=n[col].rolling(w,min_periods=w).mean(); std=n[col].rolling(w,min_periods=w).std()
            n[f'{pre}_ratio_mean_{w}']=safe_ratio(n[col],mean); n[f'{pre}_cv_{w}']=safe_ratio(std,mean.abs())
        for lag in (1,2,4,8,13,26,52): n[f'{pre}_change_{lag}']=n[col].pct_change(lag)
        n[f'{pre}_yoy_log_ratio']=np.log(n[col]/n[col].shift(52).replace(0,np.nan))
    ncols=[c for c in n.columns if c not in {'week_start','nvalue','ncount'}]
    dn=d[['week_start']].merge(n[['week_start']+ncols],on='week_start',how='left',validate='many_to_one')
    F=pd.concat([F,dn[ncols]],axis=1)

    gt=d.groupby('city',sort=False)
    for w in (13,26,52,104):
        F[f'city_decline_rate_{w}']=gt.target_float.transform(lambda s,w=w:s.shift(1).rolling(w,min_periods=max(6,min(w,13))).mean())
    weekly_rate=d.groupby('week_start',as_index=False).target_float.mean().rename(columns={'target_float':'market_decline_rate'})
    for w in (4,13,26,52):
        weekly_rate[f'market_decline_rate_{w}']=weekly_rate.market_decline_rate.shift(1).rolling(w,min_periods=max(3,min(w,8))).mean()
    rcols=[c for c in weekly_rate.columns if c not in {'week_start','market_decline_rate'}]
    dr=d[['week_start']].merge(weekly_rate[['week_start']+rcols],on='week_start',how='left',validate='many_to_one')
    F=pd.concat([F,dr[rcols]],axis=1)

    F['current_value_vs_baseline4']=safe_ratio(d.value_thousand_sar,d.baseline4)
    week=d.week_start.dt.isocalendar().week.astype(float); F['week_sin']=np.sin(2*np.pi*week/52.18); F['week_cos']=np.cos(2*np.pi*week/52.18)
    F=pd.concat([F,pd.get_dummies(d.city,prefix='city',dtype=float)],axis=1).replace([np.inf,-np.inf],np.nan)
    return d,F


def _expected_cities(features:list[str]) -> set[str]:
    """Extract only one-hot city identity columns from the frozen feature contract.

    Do NOT treat engineered features such as city_decline_rate_13 as city identities.
    City dummy suffixes in this model are uppercase canonical city codes.
    """
    cities=set()
    for c in features:
        if not c.startswith('city_') or c.startswith('city_decline_rate_'):
            continue
        suffix=c.removeprefix('city_')
        if suffix and suffix.upper()==suffix and not any(ch.isdigit() for ch in suffix):
            cities.add(suffix)
    return cities


def validate_latest_source(d:pd.DataFrame,features:list[str]) -> dict[str,Any]:
    latest=pd.Timestamp(d.week_start.max()); q=d[d.week_start==latest]
    expected=_expected_cities(features); actual=set(q.city.astype(str))
    checks={
        'nonempty_expected_city_contract':len(expected)>0,
        'latest_has_exact_expected_cities':actual==expected,
        'one_row_per_latest_city':not q.duplicated('city').any() and len(q)==len(expected),
        'positive_latest_values':bool((q.value_thousand_sar>0).all()),
        'positive_latest_counts':bool((q.transaction_count_thousand>0).all()),
        'at_least_104_weeks_history_per_city':bool(d.groupby('city').size().min()>=104),
    }
    return {'latest_week':latest,'expected_cities':sorted(expected),'actual_cities':sorted(actual),'checks':checks,'passed':all(checks.values())}


def predict_latest(panel:pd.DataFrame,base_path:Path=BASE_MODEL,policy_path:Path=POLICY_MODEL) -> dict[str,Any]:
    base=joblib.load(base_path); policy=joblib.load(policy_path)
    d,F=build_inference_features(panel)
    qc=validate_latest_source(d,base['features'])
    if not qc['passed']:
        return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'SOURCE_QC_FAILED','source_qc':qc}

    missing=[c for c in base['features'] if c not in F.columns]
    if missing: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'FEATURE_SCHEMA_MISSING:{missing}'}
    latest=qc['latest_week']; mask=d.week_start.eq(latest); X=F.loc[mask,base['features']].copy(); meta=d.loc[mask,['week_start','city']].reset_index(drop=True); X=X.reset_index(drop=True)
    bad=X.columns[X.isna().any()].tolist()
    if bad: return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':f'LATEST_FEATURE_NULLS:{bad}'}

    raw=base['model'].predict_proba(X)[:,1]
    bp=base['calibrator'].predict_proba(pd.DataFrame({base['selected']:raw}))[:,1]

    history=d[(d.week_start<latest)&d.target_float.notna()][['week_start','city','target']].rename(columns={'target':'y'}).copy()
    if len(history)<500 or history.y.sum()<20:
        return {'engine_version':ENGINE_VERSION,'status':'NO_DECISION','state':'AMBER','reason':'INSUFFICIENT_REALIZED_CALIBRATION_HISTORY'}

    out=[]
    for i,r in meta.iterrows():
        gp,cp,ep=prior.estimate_prior(history,latest,r.city,float(policy['reference_prior']))
        ap=float(prior.odds_shift([bp[i]],float(policy['reference_prior']),[ep])[0])
        state='RED' if ap>=float(policy['red_threshold']) else ('AMBER' if ap>=float(policy['watch_threshold']) else 'GREEN')
        out.append({'week_start':str(latest.date()),'city':r.city,'base_probability':float(bp[i]),'adjusted_probability':ap,'global_prior':gp,'city_prior':cp,'effective_prior':ep,'state':state})
    return {'engine_version':ENGINE_VERSION,'base_model_version':base['version'],'policy_version':policy['version'],'status':'OK','source_qc':qc,'predictions':out,'safety':{'future_label_used':False,'latest_unknown_target_required':False,'schema_fail_closed_to_amber':True,'insufficient_history_fail_closed_to_amber':True}}


def main():
    default=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
    panel=pd.read_csv(default,parse_dates=['week_start','week_end'])
    print(json.dumps(predict_latest(panel),indent=2,default=str))

if __name__=='__main__': main()
