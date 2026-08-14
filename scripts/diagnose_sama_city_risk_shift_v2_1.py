from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_sama_city_risk_v2_1 as city
import run_sama_city_risk_v2_1 as reconciled

ROOT=Path(__file__).resolve().parents[1]
HIST=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
EXT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
FRESH=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2025_2026_holdout.csv'
MODEL=ROOT/'models'/'sama_city_v2_1'/'city_market_risk_v2_1.joblib'
OUT=ROOT/'reports'/'sama_city_v2_1'/'shift_diagnostics.json'


def score(artifact,d,X):
    name=artifact['selected']; kind=artifact['selected_kind']
    if kind=='classifier': raw=artifact['models'][name].predict_proba(X)[:,1]
    elif kind=='regressor':
        pred=np.expm1(artifact['models'][name].predict(X)); ratio=pred/d.baseline4.to_numpy(); raw=city.sigmoid(((1-city.DECLINE)-ratio)/.055)
    elif name=='CurrentWeekRule':
        ratio=d.value_thousand_sar.to_numpy()/d.baseline4.to_numpy(); raw=city.sigmoid(((1-city.DECLINE)-ratio)/.055)
    else: raise RuntimeError(f'Unsupported {name}/{kind}')
    return artifact['calibrator'].predict_proba(pd.DataFrame({name:raw}))[:,1]


def city_stats(d, p, wt, rt):
    q=d[['week_start','city','target']].copy(); q['p']=p
    q['state']=np.where(q.p>=rt,'RED',np.where(q.p>=wt,'AMBER','GREEN'))
    rows=[]
    for c,g in q.groupby('city'):
        red=g.state.eq('RED'); amber=g.state.eq('AMBER'); green=g.state.eq('GREEN'); pos=g.target.eq(1); neg=~pos
        rows.append({
            'city':c,'rows':int(len(g)),'declines':int(pos.sum()),'decline_rate':float(pos.mean()),
            'red_rows':int(red.sum()),'red_tp':int((red&pos).sum()),'red_fp':int((red&neg).sum()),
            'amber_rows':int(amber.sum()),'amber_tp':int((amber&pos).sum()),'amber_fp':int((amber&neg).sum()),
            'green_rows':int(green.sum()),'green_fn':int((green&pos).sum()),
            'negative_score_p95':float(g.loc[neg,'p'].quantile(.95)) if neg.any() else None,
            'negative_score_p99':float(g.loc[neg,'p'].quantile(.99)) if neg.any() else None,
            'positive_score_median':float(g.loc[pos,'p'].median()) if pos.any() else None,
            'positive_score_min':float(g.loc[pos,'p'].min()) if pos.any() else None,
        })
    return rows,q


def feature_drift(Xa,Xb,features):
    out=[]
    for col in features:
        a=Xa[col].astype(float); b=Xb[col].astype(float)
        sd=float(a.std(ddof=0)); diff=float(b.mean()-a.mean()); smd=diff/(sd if sd>1e-12 else 1.0)
        out.append({'feature':col,'dev_mean':float(a.mean()),'fresh_mean':float(b.mean()),'dev_std':sd,'standardized_mean_shift':smd,'abs_shift':abs(smd)})
    return sorted(out,key=lambda r:r['abs_shift'],reverse=True)


def main():
    art=joblib.load(MODEL); wt=float(art['watch_threshold']); rt=float(art['red_threshold'])
    # Historical development representation.
    dh,Xh=city.featureize(reconciled.reconciled_load_panel(HIST)); keep=dh.week_start<=pd.Timestamp(art['development_end']); dh=dh.loc[keep].reset_index(drop=True); Xh=Xh.loc[keep].reset_index(drop=True)
    # Extended representation; evaluate only rows whose origin and next observation are fresh official holdout rows.
    de,Xe=city.featureize(reconciled.reconciled_load_panel(EXT)); fresh=pd.read_csv(FRESH,parse_dates=['week_start']); fsets={c:set(q.week_start.dt.normalize()) for c,q in fresh.groupby('city')}
    mask=[]
    for w,c in zip(de.week_start,de.city):
        ww=pd.Timestamp(w).normalize(); weeks=sorted(fsets.get(c,set())); mask.append(bool(weeks) and ww in fsets[c] and ww!=weeks[-1])
    df=de.loc[mask].reset_index(drop=True); Xf=Xe.loc[mask].reset_index(drop=True)
    if len(df)<400: raise RuntimeError(f'Too few fresh rows: {len(df)}')
    ph=score(art,dh,Xh); pf=score(art,df,Xf)
    hist_by_city,_=city_stats(dh,ph,wt,rt); fresh_by_city,fresh_rows=city_stats(df,pf,wt,rt)
    hmap={r['city']:r for r in hist_by_city}; comparisons=[]
    for f in fresh_by_city:
        h=hmap[f['city']]
        comparisons.append({
            'city':f['city'],'development_decline_rate':h['decline_rate'],'fresh_decline_rate':f['decline_rate'],
            'rate_ratio_fresh_to_development':f['decline_rate']/h['decline_rate'] if h['decline_rate']>0 else None,
            'fresh_red_tp':f['red_tp'],'fresh_red_fp':f['red_fp'],'fresh_green_fn':f['green_fn'],
            'fresh_negative_score_p99':f['negative_score_p99'],'fresh_positive_score_median':f['positive_score_median'],
        })
    # Coefficients of the selected logistic model in standardized feature space.
    coefficients=[]
    if art['selected']=='Logistic' and art['selected_kind']=='classifier':
        pipe=art['models']['Logistic']; clf=pipe[-1]
        for name,coef in zip(art['features'],clf.coef_[0]): coefficients.append({'feature':name,'coefficient':float(coef),'abs_coefficient':abs(float(coef))})
        coefficients=sorted(coefficients,key=lambda r:r['abs_coefficient'],reverse=True)
    report={
        'version':'SAMA-CITY-RISK-2.1-SHIFT-DIAGNOSIS',
        'diagnostic_only_not_used_to_retune_consumed_holdout':True,
        'frozen_thresholds':{'watch':wt,'red':rt},
        'development':{'rows':int(len(dh)),'declines':int(dh.target.sum()),'decline_rate':float(dh.target.mean()),'score_mean':float(np.mean(ph)),'score_p95':float(np.quantile(ph,.95))},
        'fresh':{'rows':int(len(df)),'declines':int(df.target.sum()),'decline_rate':float(df.target.mean()),'score_mean':float(np.mean(pf)),'score_p95':float(np.quantile(pf,.95))},
        'prevalence_ratio_fresh_to_development':float(df.target.mean()/dh.target.mean()),
        'city_comparison':comparisons,
        'fresh_false_positive_cities':sorted([{'city':r['city'],'red_fp':r['red_fp'],'red_tp':r['red_tp'],'declines':r['declines']} for r in fresh_by_city],key=lambda r:r['red_fp'],reverse=True),
        'top_feature_distribution_shifts':feature_drift(Xh,Xf,art['features'])[:30],
        'top_logistic_coefficients':coefficients[:30],
        'interpretation':{
            'prevalence_shift_present':bool(df.target.mean()<.75*dh.target.mean()),
            'ranking_model_can_remain_useful_while_precision_drops_when_base_rate_falls':True,
            'fresh_labels_must_not_be_used_to_select_replacement_thresholds':True,
        }
    }
    OUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
