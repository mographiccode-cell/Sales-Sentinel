from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from production_city_risk_engine_v3_4 import predict_latest
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'; OUT=ROOT/'reports'/'sama_city_v3_4'; OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2025-07-13'); END=pd.Timestamp('2026-07-26')
def m(y,p):
 y=np.asarray(y,int);p=np.asarray(p,bool);tp=int(((y==1)&p).sum());fp=int(((y==0)&p).sum());fn=int(((y==1)&~p).sum());tn=int(((y==0)&~p).sum());return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}
def main():
 d=pd.read_csv(DATA,parse_dates=['week_start','week_end']).sort_values(['city','week_start']).reset_index(drop=True);g=d.groupby('city',sort=False);d['baseline4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean());d['next_value']=g.value_thousand_sar.shift(-1);d['future_ratio']=d.next_value/d.baseline4.replace(0,np.nan);d['target']=(d.future_ratio<.8).astype('Int64');d.loc[d.future_ratio.isna(),'target']=pd.NA
 rows=[]
 for origin in sorted(d.loc[d.week_start.between(START,END),'week_start'].unique()):
  hist=d[d.week_start<=origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']];r=predict_latest(hist)
  if r.get('status')!='OK':raise RuntimeError(json.dumps(r))
  for p in r['predictions']:rows.append({'week_start':pd.Timestamp(p['week_start']),'city':p['city'],'state':p['state'],'reason':p['reason'],'risk_score':p['risk_score'],'ood_fraction':p['ood_fraction'],'precursor_count':p['precursor_count'],'structural_core_count':p['structural_core_count']})
 ev=pd.DataFrame(rows).merge(d[['week_start','city','target','future_ratio']],on=['week_start','city'],how='left',validate='one_to_one');ev=ev[ev.target.notna()].copy();ev['target']=ev.target.astype(int);y=ev.target.to_numpy();red=ev.state.eq('RED').to_numpy();alert=ev.state.isin(['RED','AMBER']).to_numpy();rm=m(y,red);am=m(y,alert)
 rep={'version':'SAMA-CITY-V3.4-RECENT-STRESS-1','independence_status':'NOT INDEPENDENT: this era informed v3 architecture; v3.4 weights/policy are not fitted here.','rows':len(ev),'weeks':int(ev.week_start.nunique()),'declines':int(y.sum()),'states':{k:int(v) for k,v in ev.state.value_counts().to_dict().items()},'alert_rate':float(alert.mean()),'green_coverage':float((~alert).mean()),'RED':rm,'RED_plus_AMBER':am,'GREEN':{'NPV':am['NPV'],'missed_declines':am['FN']},'ranking':{'ROC_AUC':float(roc_auc_score(y,ev.risk_score)),'PR_AUC':float(average_precision_score(y,ev.risk_score))},'OOD':{'rows':int(ev.reason.eq('OOD_ABSTAIN').sum()),'rate':float(ev.reason.eq('OOD_ABSTAIN').mean())},'structural_channel':{'alerts':int(ev.reason.eq('STRUCTURAL_TREND_WARNING').sum()),'positives':int(ev.loc[ev.reason.eq('STRUCTURAL_TREND_WARNING'),'target'].sum())}}
 (OUT/'recent_stress_report.json').write_text(json.dumps(rep,indent=2),encoding='utf-8');ev.to_csv(OUT/'recent_stress_predictions.csv',index=False);print(json.dumps(rep,indent=2))
if __name__=='__main__':main()
