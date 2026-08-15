from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from production_city_risk_engine_v3 import predict_latest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'saudi_synthetic_v3'
FIX = ROOT / 'tests' / 'fixtures' / 'saudi_synthetic_v3'
OUT.mkdir(parents=True, exist_ok=True)
FIX.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260815)

CITIES = {
    'RIYADH': 4_900_000.0,'JEDDAH': 3_550_000.0,'MAKKAH': 2_050_000.0,'DAMMAM': 1_650_000.0,
    'MADINA': 1_350_000.0,'KHOBAR': 1_000_000.0,'ABHA': 820_000.0,'BURAIDAH': 720_000.0,
    'HAIL': 590_000.0,'TABOUK': 560_000.0,'OTHER': 2_900_000.0,
}
TICKET = {'RIYADH':190,'JEDDAH':185,'MAKKAH':170,'DAMMAM':195,'MADINA':165,'KHOBAR':205,'ABHA':160,'BURAIDAH':155,'HAIL':150,'TABOUK':158,'OTHER':165}

# Event dates are frozen before scoring. Predictable events contain 3 completed weeks of leading deterioration.
PREDICTABLE = {
    ('2025-03-16','RIYADH'),('2025-05-18','JEDDAH'),('2025-07-20','MAKKAH'),('2025-09-21','DAMMAM'),
    ('2025-11-23','HAIL'),('2026-01-18','ABHA'),('2026-03-22','MADINA'),('2026-04-19','OTHER'),
    ('2026-05-17','BURAIDAH'),('2026-06-21','TABOUK'),('2026-07-19','KHOBAR'),
}
# Surprise events contain no precursor by construction and are reported separately, not used as a fairness gate.
SURPRISE = {('2025-04-27','ABHA'),('2025-08-31','RIYADH'),('2025-12-28','JEDDAH'),('2026-02-22','MAKKAH'),('2026-06-07','MADINA')}
POSITIVE = {('2025-06-08','MAKKAH'),('2025-09-28','RIYADH'),('2026-02-15','JEDDAH'),('2026-05-24','MADINA')}


def salary_week(ts):
    return any(d.day in (26,27,28) for d in pd.date_range(ts, ts+pd.Timedelta(days=6), freq='D'))

def season(city, ts):
    w = float(ts.isocalendar().week)
    m = 1 + .04*math.sin(2*math.pi*(w-5)/52.18)
    if salary_week(ts): m *= 1.045
    if ts.month in (3,4): m *= 1.035
    if city in {'MAKKAH','MADINA'} and ts.month in (3,4,5,6): m *= 1.05
    if city == 'ABHA' and ts.month in (6,7,8): m *= 1.08
    if ts.month == 9: m *= 1.025
    return m


def generate():
    weeks = pd.date_range('2022-01-02','2026-08-02',freq='W-SUN')
    predictable_map = {pd.Timestamp(d):c for d,c in PREDICTABLE}
    rows=[]
    state={c:1.0 for c in CITIES}
    for wi,ws in enumerate(weeks):
        shared=float(np.exp(RNG.normal(0,.012)))
        growth=1.0015**wi
        for city,base in CITIES.items():
            state[city]=float(np.clip(.99*state[city]+.01+RNG.normal(0,.0035),.95,1.05))
            val=base*growth*shared*state[city]*float(np.exp(RNG.normal(0,.018)))*season(city,ws)
            scenario='normal'
            # Leading deterioration during the 3 completed weeks BEFORE the actual event week.
            for event_date,event_city in predictable_map.items():
                if city != event_city: continue
                delta=(event_date-ws).days//7
                if delta==3: val*=.95; scenario='predictable_precursor_3'
                elif delta==2: val*=.90; scenario='predictable_precursor_2'
                elif delta==1: val*=.84; scenario='predictable_precursor_1'
                elif delta==0: val*=.64; scenario='predictable_decline'
            key=(str(ws.date()),city)
            if key in SURPRISE:
                val*=.64; scenario='surprise_decline'
            if key in POSITIVE:
                val*=1.22; scenario='positive_spike'
            ticket=TICKET[city]*(1+.01*math.sin(2*math.pi*wi/52.18))
            cnt=max(1.0,val/ticket)*float(np.exp(RNG.normal(0,.008)))
            rows.append({'week_start':ws,'week_end':ws+pd.Timedelta(days=6),'city':city,'value_thousand_sar':round(val,3),'transaction_count_thousand':round(cnt,3),'scenario':scenario})
    return pd.DataFrame(rows).sort_values(['week_start','city']).reset_index(drop=True)


def truth(panel):
    d=panel.copy().sort_values(['city','week_start']).reset_index(drop=True); g=d.groupby('city',sort=False)
    d['baseline4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
    d['next_value']=g.value_thousand_sar.shift(-1); d['next_ratio']=d.next_value/d.baseline4.replace(0,np.nan)
    d['target']=(d.next_ratio<.80).astype('Int64'); d.loc[d.next_ratio.isna(),'target']=pd.NA
    next_scenario=g.scenario.shift(-1); d['next_scenario']=next_scenario
    d['event_type']=np.where(next_scenario.eq('predictable_decline'),'FORECASTABLE',np.where(next_scenario.eq('surprise_decline'),'SURPRISE','OTHER'))
    return d


def metric(y,p):
    y=np.asarray(y,int); p=np.asarray(p,bool)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&~p).sum()); tn=int(((y==0)&~p).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}


def main():
    panel=generate(); gt=truth(panel)
    pred=[]
    weeks=sorted(panel.week_start.unique())
    # Start after 112 weeks so all rolling features are warm and history contract is met.
    for origin in weeks[112:-1]:
        hist=panel[panel.week_start<=origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']]
        r=predict_latest(hist)
        if r.get('status')!='OK':
            raise RuntimeError(json.dumps(r))
        for p in r['predictions']:
            pred.append({'week_start':pd.Timestamp(p['week_start']),'city':p['city'],'state':p['state'],'reason':p['reason'],'risk_score':p['risk_score'],'precursor_count':p['precursor_count'],'ood_fraction':p['ood_fraction']})
    pr=pd.DataFrame(pred)
    ev=pr.merge(gt[['week_start','city','scenario','next_scenario','event_type','target','next_ratio']],on=['week_start','city'],how='left',validate='one_to_one')
    ev=ev[ev.target.notna()].copy(); ev['target']=ev.target.astype(int)
    red=ev.state.eq('RED'); alert=ev.state.isin(['RED','AMBER']); y=ev.target
    all_red=metric(y,red); all_alert=metric(y,alert)
    forecastable=ev.event_type.eq('FORECASTABLE')
    surprise=ev.event_type.eq('SURPRISE')
    normal=~ev.event_type.isin(['FORECASTABLE','SURPRISE'])
    report={
        'version':'SAUDI-SYNTHETIC-V3-FAIR-ROBUSTNESS-1','seed':20260815,'synthetic_not_official':True,
        'rows_scored':int(len(ev)),'weeks_scored':int(ev.week_start.nunique()),'actual_declines':int(y.sum()),
        'states':{k:int(v) for k,v in ev.state.value_counts().to_dict().items()},
        'all_actual_declines':{'RED':all_red,'RED_plus_AMBER':all_alert},
        'forecastable_events':{
            'origins':int(forecastable.sum()),'actual_declines':int(ev.loc[forecastable,'target'].sum()),
            'alerted':int((alert&forecastable&y.eq(1)).sum()),
            'alert_recall':float((alert&forecastable&y.eq(1)).sum()/max(int((forecastable&y.eq(1)).sum()),1)),
            'red':int((red&forecastable&y.eq(1)).sum()),
        },
        'surprise_events':{
            'origins':int(surprise.sum()),'actual_declines':int(ev.loc[surprise,'target'].sum()),
            'alerted':int((alert&surprise&y.eq(1)).sum()),
            'note':'No leading signal was provided by construction; pre-event detection is not an acceptance gate.'
        },
        'normal_origins':{
            'rows':int(normal.sum()),'false_red':int((red&normal&y.eq(0)).sum()),
            'false_red_rate':float((red&normal&y.eq(0)).sum()/max(int((normal&y.eq(0)).sum()),1)),
        },
        'ood':{'amber_abstentions':int(ev.reason.eq('OOD_ABSTAIN').sum()),'rate':float(ev.reason.eq('OOD_ABSTAIN').mean())},
        'acceptance':{
            'forecastable_alert_recall_ge_80pct':bool((alert&forecastable&y.eq(1)).sum()/max(int((forecastable&y.eq(1)).sum()),1)>=.80),
            'normal_false_red_rate_le_3pct':bool((red&normal&y.eq(0)).sum()/max(int((normal&y.eq(0)).sum()),1)<=.03),
            'all_red_precision_ge_60pct':bool(all_red['precision']>=.60),
        },
    }
    report['all_acceptance_passed']=bool(all(report['acceptance'].values()))
    FIX.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
    panel[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']].to_csv(FIX/'saudi_v3_app_input.csv',index=False)
    gt[['week_start','week_end','city','scenario','next_scenario','event_type','target','next_ratio']].to_csv(FIX/'saudi_v3_ground_truth.csv',index=False)
    ev.to_csv(FIX/'saudi_v3_predictions.csv',index=False)
    (OUT/'synthetic_v3_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
