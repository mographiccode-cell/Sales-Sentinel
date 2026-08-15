from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from production_city_risk_engine_v3_4 import predict_latest
import train_sama_city_risk_v3 as feature_source

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
OUT=ROOT/'reports'/'sama_city_v3_4'; FIX=ROOT/'tests'/'fixtures'/'sama_city_v3_4_fresh_acceptance'
OUT.mkdir(parents=True,exist_ok=True); FIX.mkdir(parents=True,exist_ok=True)
SEED=3429; RNG=np.random.default_rng(SEED)
TEST_START=pd.Timestamp('2025-09-28'); TEST_END=pd.Timestamp('2026-07-12')
# Previously exposed event keys are forbidden so this acceptance set is genuinely new relative to v3.3 diagnostics.
PREVIOUS={
 ('2025-09-14','HAIL'),('2025-10-12','BURAIDAH'),('2025-11-09','TABOUK'),('2025-12-07','OTHER'),('2026-01-04','ABHA'),('2026-02-01','MADINA'),('2026-03-01','RIYADH'),('2026-03-29','KHOBAR'),('2026-04-26','JEDDAH'),('2026-05-24','MAKKAH'),('2026-06-21','DAMMAM')
}


def truth(d):
    z=d.copy().sort_values(['city','week_start']).reset_index(drop=True); g=z.groupby('city',sort=False)
    z['baseline4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean()); z['next_value']=g.value_thousand_sar.shift(-1); z['future_ratio']=z.next_value/z.baseline4.replace(0,np.nan); z['target']=(z.future_ratio<.80).astype('Int64'); z.loc[z.future_ratio.isna(),'target']=pd.NA
    return z


def schedule(base):
    cities=sorted(base.city.astype(str).unique()); RNG.shuffle(cities)
    dates=[TEST_START+pd.Timedelta(weeks=4*i) for i in range(len(cities))]
    pairs=list(zip(dates,cities))
    if any((str(o.date()),c) in PREVIOUS for o,c in pairs): raise RuntimeError('Fresh acceptance accidentally overlaps exposed event key')
    if dates[-1]+pd.Timedelta(weeks=1)>TEST_END: raise RuntimeError('Insufficient acceptance horizon')
    return pairs


def inject(base,events):
    d=base.copy(); d['injection']='none'; paths=[]
    for origin,city in events:
        anchor_ws=origin-pd.Timedelta(weeks=3); a=d[d.city.eq(city)&d.week_start.eq(anchor_ws)]
        if len(a)!=1: raise RuntimeError('Missing anchor')
        av=float(a.iloc[0].value_thousand_sar); ac=float(a.iloc[0].transaction_count_thousand)
        # Event severity varies deterministically by seed; the model never sees these parameters during training.
        p2=float(RNG.uniform(.955,.975)); p1=float(RNG.uniform(.875,.910)); p0=float(RNG.uniform(.790,.825)); pnext=float(RNG.uniform(.565,.625))
        c2=float(RNG.uniform(.960,.980)); c1=float(RNG.uniform(.880,.915)); c0=float(RNG.uniform(.795,.830)); cnext=float(RNG.uniform(.575,.635))
        vp={-2:p2,-1:p1,0:p0,1:pnext}; cp={-2:c2,-1:c1,0:c0,1:cnext}
        paths.append({'origin':str(origin.date()),'city':city,'value_path':vp,'count_path':cp})
        for off in (-2,-1,0,1):
            ws=origin+pd.Timedelta(weeks=off); m=d.city.eq(city)&d.week_start.eq(ws)
            if int(m.sum())!=1: raise RuntimeError('Missing injection row')
            d.loc[m,'value_thousand_sar']=av*vp[off]; d.loc[m,'transaction_count_thousand']=ac*cp[off]; d.loc[m,'injection']=f'fresh_origin_{origin.date()}_offset_{off}'
    return d,paths


def validate_observable(modified,events):
    d,X,P,pc=feature_source.featureize(modified[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']],require_target=False)
    core=['value_below_short_trend','count_below_short_trend','value_negative_slope','count_negative_slope','value_2w_drop','count_2w_drop']
    rows=[]
    for origin,city in events:
        m=d.week_start.eq(origin)&d.city.eq(city)
        if int(m.sum())!=1: raise RuntimeError('Missing event features')
        i=d.index[m][0]; core_count=int(P.loc[i,core].sum()); rows.append({'origin':origin,'city':city,'core_count':core_count,'precursor_count':int(pc.loc[i])})
    q=pd.DataFrame(rows)
    if (q.core_count<6).any(): raise RuntimeError('Acceptance construction invalid: all events must satisfy all six structural trend signals')
    return q


def main():
    base=pd.read_csv(DATA,parse_dates=['week_start','week_end']).sort_values(['week_start','city']).reset_index(drop=True)
    events=schedule(base); modified,paths=inject(base,events); obs=validate_observable(modified,events); gt=truth(modified); keys={(o,c) for o,c in events}
    intended=gt[[((w,c) in keys) for w,c in zip(gt.week_start,gt.city)]].copy()
    if len(intended)!=len(events) or not intended.target.eq(1).all(): raise RuntimeError('Acceptance construction invalid: every origin must create actual >=20% next-week decline')

    rows=[]
    for origin in sorted(modified.loc[modified.week_start.between(TEST_START,TEST_END),'week_start'].unique()):
        hist=modified[modified.week_start<=origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']]; r=predict_latest(hist)
        if r.get('status')!='OK': raise RuntimeError(json.dumps(r))
        for p in r['predictions']: rows.append({'week_start':pd.Timestamp(p['week_start']),'city':p['city'],'state':p['state'],'reason':p['reason'],'risk_score':p['risk_score'],'precursor_count':p['precursor_count'],'structural_core_count':p['structural_core_count'],'ood_fraction':p['ood_fraction']})
    ev=pd.DataFrame(rows).merge(gt[['week_start','city','target','future_ratio']],on=['week_start','city'],how='left',validate='one_to_one'); ev=ev[ev.target.notna()].copy(); ev['target']=ev.target.astype(int); ev['is_injected_event_origin']=[(w,c) in keys for w,c in zip(ev.week_start,ev.city)]
    contaminated=np.zeros(len(ev),dtype=bool)
    for origin,city in events: contaminated|=(ev.city.eq(city)&ev.week_start.between(origin-pd.Timedelta(weeks=3),origin+pd.Timedelta(weeks=3))).to_numpy()
    ev['is_clean_control']=~contaminated
    alert=ev.state.isin(['RED','AMBER']); red=ev.state.eq('RED'); inj=ev.is_injected_event_origin&ev.target.eq(1); ctrl=ev.is_clean_control; ctrl_neg=ctrl&ev.target.eq(0)
    inj_recall=float((alert&inj).sum()/max(int(inj.sum()),1)); ctrl_fr=int((red&ctrl_neg).sum()); ctrl_fr_rate=float(ctrl_fr/max(int(ctrl_neg.sum()),1)); ctrl_alert=float((alert&ctrl).sum()/max(int(ctrl.sum()),1)); ctrl_green=1-ctrl_alert
    report={'version':'SAMA-CITY-V3.4-FRESH-ACCEPTANCE-1','seed':SEED,'frozen_model':'SAMA-CITY-RISK-3.4-STRUCTURAL-HYBRID','freshness':'No event key overlaps the corrected v3.3 diagnostic set; event severities are newly sampled and model-score blind.','events':[{'origin':str(o.date()),'city':c} for o,c in events],'paths':paths,'observability':[{'origin':str(r.origin.date()),'city':r.city,'core_count':int(r.core_count),'precursor_count':int(r.precursor_count)} for r in obs.itertuples(index=False)],'event_count':len(events),'rows_scored':int(len(ev)),'injected_event_actual_declines':int(inj.sum()),'injected_event_alerted':int((alert&inj).sum()),'injected_event_alert_recall':inj_recall,'injected_event_red':int((red&inj).sum()),'injected_event_structural_alerts':int((ev.reason.eq('STRUCTURAL_TREND_WARNING')&inj).sum()),'control':{'rows':int(ctrl.sum()),'negative_rows':int(ctrl_neg.sum()),'false_red':ctrl_fr,'false_red_rate':ctrl_fr_rate,'alert_rate':ctrl_alert,'green_coverage':ctrl_green},'overall':{'states':{k:int(v) for k,v in ev.state.value_counts().to_dict().items()},'OOD_abstentions':int(ev.reason.eq('OOD_ABSTAIN').sum()),'structural_alerts':int(ev.reason.eq('STRUCTURAL_TREND_WARNING').sum())},'acceptance':{'test_core6_valid':bool((obs.core_count==6).all()),'all_injected_origins_actual_declines':bool(int(inj.sum())==len(events)),'injected_alert_recall_ge_90pct':bool(inj_recall>=.90),'control_false_red_rate_le_2pct':bool(ctrl_fr_rate<=.02),'control_alert_rate_le_35pct':bool(ctrl_alert<=.35),'control_green_coverage_ge_65pct':bool(ctrl_green>=.65)}}
    report['all_acceptance_passed']=bool(all(report['acceptance'].values()))
    modified[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand','injection']].to_csv(FIX/'fresh_acceptance_input.csv',index=False); ev.to_csv(FIX/'fresh_acceptance_predictions_and_truth.csv',index=False); (OUT/'fresh_acceptance_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
