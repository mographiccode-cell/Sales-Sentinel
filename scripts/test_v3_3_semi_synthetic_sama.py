from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from production_city_risk_engine_v3_3 import predict_latest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
OUT=ROOT/'reports'/'sama_city_v3_3'; FIX=ROOT/'tests'/'fixtures'/'sama_city_v3_3_semi_synthetic'
OUT.mkdir(parents=True,exist_ok=True); FIX.mkdir(parents=True,exist_ok=True)
SEED=3303; RNG=np.random.default_rng(SEED)
TEST_START=pd.Timestamp('2025-09-07'); TEST_END=pd.Timestamp('2026-05-31')


def metrics(y,p):
 y=np.asarray(y,int);p=np.asarray(p,bool);tp=int(((y==1)&p).sum());fp=int(((y==0)&p).sum());fn=int(((y==1)&~p).sum());tn=int(((y==0)&~p).sum());return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}

def build_truth(d):
 z=d.copy().sort_values(['city','week_start']).reset_index(drop=True);g=z.groupby('city',sort=False);z['baseline4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean());z['next_value']=g.value_thousand_sar.shift(-1);z['future_ratio']=z.next_value/z.baseline4.replace(0,np.nan);z['target']=(z.future_ratio<.8).astype('Int64');z.loc[z.future_ratio.isna(),'target']=pd.NA;return z

def choose_events(base):
 truth=build_truth(base); cand=truth[truth.week_start.between(TEST_START+pd.Timedelta(weeks=4),TEST_END-pd.Timedelta(weeks=2)) & truth.target.eq(0) & truth.future_ratio.between(.90,1.15)].copy()
 selected=[]
 for city in sorted(cand.city.unique()):
  c=cand[cand.city.eq(city)].sort_values('week_start')
  if c.empty: continue
  # Deterministic random selection independent of any model score.
  idx=int(RNG.integers(0,len(c))); selected.append((pd.Timestamp(c.iloc[idx].week_start),city))
 return selected

def inject(base,events):
 d=base.copy(); d['injection']='none'
 # Real SAMA shape retained. For each event origin t, create gradual deterioration in t-2,t-1,t and a strong decline in t+1.
 multipliers={-2:.97,-1:.91,0:.83,1:.61}
 for origin,city in events:
  for offset,mul in multipliers.items():
   ws=origin+pd.Timedelta(weeks=offset); mask=d.city.eq(city)&d.week_start.eq(ws)
   if int(mask.sum())!=1: raise RuntimeError(f'Missing injection row {city} {ws.date()}')
   d.loc[mask,'value_thousand_sar']*=mul; d.loc[mask,'transaction_count_thousand']*=mul
   d.loc[mask,'injection']=f'event_origin_{origin.date()}_offset_{offset}'
 return d

def main():
 base=pd.read_csv(DATA,parse_dates=['week_start','week_end']).sort_values(['week_start','city']).reset_index(drop=True)
 events=choose_events(base); modified=inject(base,events); truth=build_truth(modified)
 rows=[]
 for origin in sorted(modified.loc[modified.week_start.between(TEST_START,TEST_END),'week_start'].unique()):
  hist=modified[modified.week_start<=origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']];r=predict_latest(hist)
  if r.get('status')!='OK':raise RuntimeError(json.dumps(r))
  for p in r['predictions']:rows.append({'week_start':pd.Timestamp(p['week_start']),'city':p['city'],'state':p['state'],'reason':p['reason'],'risk_score':p['risk_score'],'precursor_count':p['precursor_count'],'ood_fraction':p['ood_fraction']})
 ev=pd.DataFrame(rows).merge(truth[['week_start','city','target','future_ratio']],on=['week_start','city'],how='left',validate='one_to_one');ev=ev[ev.target.notna()].copy();ev['target']=ev.target.astype(int)
 event_keys={(o,c) for o,c in events};ev['is_injected_event_origin']=[(pd.Timestamp(w),c) in event_keys for w,c in zip(ev.week_start,ev.city)]
 # Controls exclude the selected city within +/-3 weeks around an injection to avoid counting precursor/recovery behavior as ordinary controls.
 contaminated=np.zeros(len(ev),dtype=bool)
 for origin,city in events: contaminated|=(ev.city.eq(city)&ev.week_start.between(origin-pd.Timedelta(weeks=3),origin+pd.Timedelta(weeks=3))).to_numpy()
 ev['is_clean_control']=~contaminated
 alert=ev.state.isin(['RED','AMBER']); red=ev.state.eq('RED'); inj=ev.is_injected_event_origin; ctrl=ev.is_clean_control
 inj_actual=inj&ev.target.eq(1); injected_alert_recall=float((alert&inj_actual).sum()/max(int(inj_actual.sum()),1)); injected_red=int((red&inj_actual).sum())
 ctrl_neg=ctrl&ev.target.eq(0); ctrl_false_red=int((red&ctrl_neg).sum()); ctrl_false_red_rate=float(ctrl_false_red/max(int(ctrl_neg.sum()),1)); ctrl_alert_rate=float((alert&ctrl).sum()/max(int(ctrl.sum()),1)); ctrl_green_coverage=1-ctrl_alert_rate
 report={'version':'SAMA-CITY-V3.3-SEMI-SYNTHETIC-FRESH-1','seed':SEED,'frozen_model':'SAMA-CITY-RISK-3.3-DUAL-CHANNEL','test_design':'Fresh controlled perturbations injected into real SAMA city-total series after v3.3 policy freeze. Event selection uses dates/base target only, never model score.','events':[{'origin':str(o.date()),'city':c} for o,c in events],'event_count':len(events),'rows_scored':len(ev),'injected_event_actual_declines':int(inj_actual.sum()),'injected_event_alerted':int((alert&inj_actual).sum()),'injected_event_alert_recall':injected_alert_recall,'injected_event_red':injected_red,'control':{'rows':int(ctrl.sum()),'negative_rows':int(ctrl_neg.sum()),'false_red':ctrl_false_red,'false_red_rate':ctrl_false_red_rate,'alert_rate':ctrl_alert_rate,'green_coverage':ctrl_green_coverage},'overall':{'states':{k:int(v) for k,v in ev.state.value_counts().to_dict().items()},'OOD_abstentions':int(ev.reason.eq('OOD_ABSTAIN').sum()),'fallback_alerts':int(ev.reason.eq('HIGH_PRECURSOR_FALLBACK').sum())},'acceptance':{'injected_alert_recall_ge_85pct':injected_alert_recall>=.85,'control_false_red_rate_le_2pct':ctrl_false_red_rate<=.02,'control_alert_rate_le_30pct':ctrl_alert_rate<=.30,'control_green_coverage_ge_70pct':ctrl_green_coverage>=.70}}
 report['all_acceptance_passed']=bool(all(report['acceptance'].values()))
 modified[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand','injection']].to_csv(FIX/'semi_synthetic_sama_input.csv',index=False);ev.to_csv(FIX/'semi_synthetic_sama_predictions_and_truth.csv',index=False);(OUT/'semi_synthetic_fresh_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
