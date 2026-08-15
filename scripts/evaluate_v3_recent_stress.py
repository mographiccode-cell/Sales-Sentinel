from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from production_city_risk_engine_v3 import predict_latest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
OUT=ROOT/'reports'/'sama_city_v3'
OUT.mkdir(parents=True,exist_ok=True)
START=pd.Timestamp('2025-07-13'); END=pd.Timestamp('2026-07-26')


def metric(y,p):
    y=np.asarray(y,int); p=np.asarray(p,bool)
    tp=int(((y==1)&p).sum()); fp=int(((y==0)&p).sum()); fn=int(((y==1)&~p).sum()); tn=int(((y==0)&~p).sum())
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'FPR':fp/max(fp+tn,1),'NPV':tn/max(tn+fn,1)}


def main():
    panel=pd.read_csv(DATA,parse_dates=['week_start','week_end']).sort_values(['city','week_start']).reset_index(drop=True)
    g=panel.groupby('city',sort=False)
    panel['baseline4']=g.value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
    panel['next_value']=g.value_thousand_sar.shift(-1)
    panel['future_ratio']=panel.next_value/panel.baseline4.replace(0,np.nan)
    panel['target']=(panel.future_ratio<.80).astype('Int64'); panel.loc[panel.future_ratio.isna(),'target']=pd.NA
    rows=[]
    for origin in sorted(panel.loc[panel.week_start.between(START,END),'week_start'].unique()):
        hist=panel[panel.week_start<=origin][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']]
        r=predict_latest(hist)
        if r.get('status')!='OK':
            raise RuntimeError(json.dumps(r))
        for p in r['predictions']:
            rows.append({'week_start':pd.Timestamp(p['week_start']),'city':p['city'],'state':p['state'],'reason':p['reason'],'risk_score':p['risk_score'],'ood_fraction':p['ood_fraction'],'precursor_count':p['precursor_count']})
    ev=pd.DataFrame(rows).merge(panel[['week_start','city','target','future_ratio']],on=['week_start','city'],how='left',validate='one_to_one')
    ev=ev[ev.target.notna()].copy(); ev['target']=ev.target.astype(int)
    y=ev.target.to_numpy(int); red=ev.state.eq('RED').to_numpy(); alert=ev.state.isin(['RED','AMBER']).to_numpy()
    redm=metric(y,red); alertm=metric(y,alert)
    report={
        'version':'SAMA-CITY-V3-RECENT-STRESS-1',
        'independence_status':'NOT INDEPENDENT: v3 architecture was designed after prior diagnosis of this era. No v3 weights or thresholds are fitted here.',
        'period':{'start':str(START.date()),'end':str(END.date())},'rows':int(len(ev)),'weeks':int(ev.week_start.nunique()),
        'declines':int(y.sum()),'decline_rate':float(y.mean()),'states':{k:int(v) for k,v in ev.state.value_counts().to_dict().items()},
        'RED':redm,'RED_plus_AMBER':alertm,'GREEN':{'NPV':alertm['NPV'],'missed_declines':alertm['FN']},
        'ranking':{'ROC_AUC':float(roc_auc_score(y,ev.risk_score)),'PR_AUC':float(average_precision_score(y,ev.risk_score))},
        'OOD':{'rows':int(ev.reason.eq('OOD_ABSTAIN').sum()),'rate':float(ev.reason.eq('OOD_ABSTAIN').mean()),'max_fraction':float(ev.ood_fraction.max())},
    }
    (OUT/'recent_stress_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    ev.to_csv(OUT/'recent_stress_predictions.csv',index=False)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
