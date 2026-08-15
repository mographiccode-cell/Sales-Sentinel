from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from production_city_risk_engine_v3_3 import predict_latest
import train_sama_city_risk_v3 as feature_source

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
OUT=ROOT/'reports'/'sama_city_v3_3'
FIX=ROOT/'tests'/'fixtures'/'sama_city_v3_3_semi_synthetic_v2'
OUT.mkdir(parents=True,exist_ok=True); FIX.mkdir(parents=True,exist_ok=True)
SEED=3311
RNG=np.random.default_rng(SEED)
TEST_START=pd.Timestamp('2025-09-14')
TEST_END=pd.Timestamp('2026-07-12')


def metrics(y,p):
    y=np.asarray(y,int); p=np.asarray(p,bool)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&~p).sum()); tn=int(((y==0)&~p).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}


def build_truth(d):
    z=d.copy().sort_values(['city','week_start']).reset_index(drop=True)
    g=z.groupby('city',sort=False)
    z['baseline4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
    z['next_value']=g.value_thousand_sar.shift(-1)
    z['future_ratio']=z.next_value/z.baseline4.replace(0,np.nan)
    z['target']=(z.future_ratio<.80).astype('Int64')
    z.loc[z.future_ratio.isna(),'target']=pd.NA
    return z


def schedule_events(base):
    cities=sorted(base.city.astype(str).unique())
    RNG.shuffle(cities)
    dates=[TEST_START+pd.Timedelta(weeks=4*i) for i in range(len(cities))]
    if dates[-1]+pd.Timedelta(weeks=1)>TEST_END:
        raise RuntimeError('Not enough test horizon for spaced events')
    return list(zip(dates,cities))


def overwrite_event_path(base,events):
    d=base.copy(); d['injection']='none'
    # Anchor at t-3; then force an actually observable monotonic deterioration at t-2,t-1,t,
    # followed by the target decline at t+1. This avoids the flaw in v1 where multiplying the
    # raw SAMA value could still leave an apparent weekly increase.
    value_path={-2:.97,-1:.90,0:.82,1:.60}
    count_path={-2:.98,-1:.91,0:.83,1:.62}
    for origin,city in events:
        anchor_ws=origin-pd.Timedelta(weeks=3)
        anchor=d[d.city.eq(city)&d.week_start.eq(anchor_ws)]
        if len(anchor)!=1: raise RuntimeError(f'Missing anchor {city} {anchor_ws.date()}')
        av=float(anchor.iloc[0].value_thousand_sar); ac=float(anchor.iloc[0].transaction_count_thousand)
        for off in (-2,-1,0,1):
            ws=origin+pd.Timedelta(weeks=off); mask=d.city.eq(city)&d.week_start.eq(ws)
            if int(mask.sum())!=1: raise RuntimeError(f'Missing event row {city} {ws.date()}')
            d.loc[mask,'value_thousand_sar']=av*value_path[off]
            d.loc[mask,'transaction_count_thousand']=ac*count_path[off]
            d.loc[mask,'injection']=f'origin_{origin.date()}_offset_{off}'
    return d


def validate_event_observability(modified,events):
    d,X,P,pc=feature_source.featureize(modified[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']],require_target=False)
    rows=[]
    for origin,city in events:
        m=d.week_start.eq(origin)&d.city.eq(city)
        if int(m.sum())!=1: raise RuntimeError(f'No feature row for event {city} {origin.date()}')
        i=d.index[m][0]
        names=[c for c in P.columns if bool(P.loc[i,c])]
        rows.append({'origin':origin,'city':city,'precursor_count':int(pc.loc[i]),'precursors':names})
    q=pd.DataFrame(rows)
    if (q.precursor_count<6).any():
        raise RuntimeError('Invalid test construction: every injected event must expose >=6/7 deterioration precursors before scoring')
    return q


def main():
    base=pd.read_csv(DATA,parse_dates=['week_start','week_end']).sort_values(['week_start','city']).reset_index(drop=True)
    events=schedule_events(base)
    modified=overwrite_event_path(base,events)
    observability=validate_event_observability(modified,events)
    truth=build_truth(modified)

    event_keys={(o,c) for o,c in events}
    # Validate that the intended origin actually produces the target decline.
    intended=truth[[((w,c) in event_keys) for w,c in zip(truth.week_start,truth.city)]].copy()
    if len(intended)!=len(events) or not intended.target.eq(1).all():
        raise RuntimeError('Invalid test construction: not every intended origin produces the >=20% next-week decline target')

    rows=[]
    for origin in sorted(modified.loc[modified.week_start.between(TEST_START,TEST_END),'week_start'].unique()):
        hist=modified[modified.week_start<=origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']]
        r=predict_latest(hist)
        if r.get('status')!='OK': raise RuntimeError(json.dumps(r))
        for p in r['predictions']:
            rows.append({'week_start':pd.Timestamp(p['week_start']),'city':p['city'],'state':p['state'],'reason':p['reason'],'risk_score':p['risk_score'],'precursor_count':p['precursor_count'],'ood_fraction':p['ood_fraction']})
    ev=pd.DataFrame(rows).merge(truth[['week_start','city','target','future_ratio']],on=['week_start','city'],how='left',validate='one_to_one')
    ev=ev[ev.target.notna()].copy(); ev['target']=ev.target.astype(int)
    ev['is_injected_event_origin']=[(w,c) in event_keys for w,c in zip(ev.week_start,ev.city)]

    # Exclude each injected city's local +/-3-week event neighborhood from clean controls.
    contaminated=np.zeros(len(ev),dtype=bool)
    for origin,city in events:
        contaminated|=(ev.city.eq(city)&ev.week_start.between(origin-pd.Timedelta(weeks=3),origin+pd.Timedelta(weeks=3))).to_numpy()
    ev['is_clean_control']=~contaminated

    alert=ev.state.isin(['RED','AMBER']); red=ev.state.eq('RED')
    inj=ev.is_injected_event_origin&ev.target.eq(1)
    ctrl=ev.is_clean_control; ctrl_neg=ctrl&ev.target.eq(0)
    inj_recall=float((alert&inj).sum()/max(int(inj.sum()),1))
    ctrl_false_red=int((red&ctrl_neg).sum()); ctrl_false_red_rate=float(ctrl_false_red/max(int(ctrl_neg.sum()),1))
    ctrl_alert_rate=float((alert&ctrl).sum()/max(int(ctrl.sum()),1)); ctrl_green=1-ctrl_alert_rate
    report={
        'version':'SAMA-CITY-V3.3-SEMI-SYNTHETIC-CORRECTED-FRESH-2',
        'seed':SEED,
        'frozen_model':'SAMA-CITY-RISK-3.3-DUAL-CHANNEL',
        'test_design':'Real SAMA city-total series with new model-score-blind event dates. Each event path is overwritten from its t-3 anchor to guarantee observable monotonic value/count deterioration before the target decline. Test validity requires >=6/7 precursor indicators at every event origin.',
        'events':[{'origin':str(o.date()),'city':c} for o,c in events],
        'observability':[
            {'origin':str(r.origin.date()),'city':r.city,'precursor_count':int(r.precursor_count),'precursors':r.precursors}
            for r in observability.itertuples(index=False)
        ],
        'event_count':len(events),'rows_scored':int(len(ev)),
        'injected_event_actual_declines':int(inj.sum()),'injected_event_alerted':int((alert&inj).sum()),
        'injected_event_alert_recall':inj_recall,'injected_event_red':int((red&inj).sum()),
        'control':{'rows':int(ctrl.sum()),'negative_rows':int(ctrl_neg.sum()),'false_red':ctrl_false_red,'false_red_rate':ctrl_false_red_rate,'alert_rate':ctrl_alert_rate,'green_coverage':ctrl_green},
        'overall':{'states':{k:int(v) for k,v in ev.state.value_counts().to_dict().items()},'OOD_abstentions':int(ev.reason.eq('OOD_ABSTAIN').sum()),'fallback_alerts':int(ev.reason.eq('HIGH_PRECURSOR_FALLBACK').sum())},
        'acceptance':{
            'test_observability_valid':bool((observability.precursor_count>=6).all()),
            'all_injected_origins_are_actual_declines':bool(int(inj.sum())==len(events)),
            'injected_alert_recall_ge_85pct':bool(inj_recall>=.85),
            'control_false_red_rate_le_2pct':bool(ctrl_false_red_rate<=.02),
            'control_alert_rate_le_35pct':bool(ctrl_alert_rate<=.35),
            'control_green_coverage_ge_65pct':bool(ctrl_green>=.65),
        },
    }
    report['all_acceptance_passed']=bool(all(report['acceptance'].values()))
    modified[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand','injection']].to_csv(FIX/'semi_synthetic_corrected_input.csv',index=False)
    ev.to_csv(FIX/'semi_synthetic_corrected_predictions_and_truth.csv',index=False)
    (OUT/'semi_synthetic_corrected_fresh_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
