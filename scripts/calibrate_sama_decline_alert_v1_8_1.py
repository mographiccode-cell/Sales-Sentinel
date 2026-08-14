from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
HIST=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2020_2025.csv'
FCST=ROOT/'data'/'sama_pos'/'sama_sector_walkforward_forecasts_2023_2025.csv'
OUT=ROOT/'reports'/'sama_sector_v1_8_1'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'calibrated_sama_decline_alert_v1_8_1.json'; SUMMARY=OUT/'calibrated_sama_decline_alert_v1_8_1.md'
TRUE_DECLINE=.20

def metric(y,score,threshold):
    pred=(score>=threshold).astype(int); y=np.asarray(y).astype(int)
    return {'Accuracy':float(accuracy_score(y,pred)),'BalancedAccuracy':float(balanced_accuracy_score(y,pred)),'Precision':float(precision_score(y,pred,zero_division=0)),'Recall':float(recall_score(y,pred,zero_division=0)),'F1':float(f1_score(y,pred,zero_division=0)),'ROC_AUC':float(roc_auc_score(y,score)),'ConfusionMatrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}
def main():
    h=pd.read_csv(HIST,parse_dates=['week_start']).sort_values(['sector','week_start']); h['baseline4']=h.groupby('sector').value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
    base=h[['week_start','sector','baseline4']].rename(columns={'week_start':'origin_week_start'})
    f=pd.read_csv(FCST,parse_dates=['origin_week_start']); d=f.merge(base,on=['origin_week_start','sector'],how='left',validate='many_to_one').dropna(subset=['baseline4','predicted_value_h1','actual_value_h1']).copy()
    d['target']=(d.actual_value_h1 < (1-TRUE_DECLINE)*d.baseline4).astype(int)
    # forecasted decline fraction: +0.20 means forecast says 20% below the known baseline.
    d['forecast_decline']=1-d.predicted_value_h1/d.baseline4
    val=d[(d.origin_week_start>='2023-01-01')&(d.origin_week_start<'2024-01-01')].copy(); test=d[d.origin_week_start>='2024-01-01'].copy()
    candidates=[]
    for t in np.arange(-.05,.351,.0025):
        m=metric(val.target.to_numpy(),val.forecast_decline.to_numpy(),float(t))
        # Accuracy must stay high on validation; among acceptable thresholds maximize F1 then recall/balanced accuracy.
        feasible=m['Accuracy']>=.90 and m['Recall']>=.45
        score=.38*m['F1']+.25*m['Recall']+.20*m['BalancedAccuracy']+.10*m['Accuracy']+.07*m['ROC_AUC']
        candidates.append((feasible,score,m['F1'],m['Recall'],m['Accuracy'],m['BalancedAccuracy'],-abs(t-.15),float(t),m))
    feasible=[c for c in candidates if c[0]]
    chosen=max(feasible,key=lambda c:c[1:7]) if feasible else max(candidates,key=lambda c:c[1:7])
    threshold=chosen[7]; vm=chosen[8]; tm=metric(test.target.to_numpy(),test.forecast_decline.to_numpy(),threshold); majority=max(float(test.target.mean()),1-float(test.target.mean()))
    gates={'accuracy_at_least_90pct':tm['Accuracy']>=.90,'balanced_accuracy_at_least_75pct':tm['BalancedAccuracy']>=.75,'recall_at_least_55pct':tm['Recall']>=.55,'f1_at_least_55pct':tm['F1']>=.55,'roc_auc_at_least_85pct':tm['ROC_AUC']>=.85,'beats_majority':tm['Accuracy']>majority}
    out={'version':'SAMA-SECTOR-DECLINE-1.8.1','true_target':'actual next official SAMA sector week >=20% below trailing four-week baseline','alert_threshold_selected_on_2023_only':threshold,'interpretation':f'Raise a decline alert when forecasted sector decline is >= {threshold:.2%}; the true event remains fixed at >=20%.','validation_2023':{'rows':len(val),'positive_rate':float(val.target.mean()),'metrics':vm},'test_2024_2025':{'rows':len(test),'positive_rate':float(test.target.mean()),'metrics':tm},'majority_test_accuracy':majority,'acceptance_gates':gates,'all_gates_passed':bool(all(gates.values())),'leakage_controls':{'target_never_changed':True,'alert_threshold_selected_only_on_2023':True,'2024_2025_not_used_for_threshold':True,'forecast_is_walk_forward':True}}
    REPORT.write_text(json.dumps(out,indent=2),encoding='utf-8'); SUMMARY.write_text(f'''# Calibrated SAMA Decline Alert v1.8.1

- True decline target: **20%** (unchanged)
- Alert threshold selected from 2023 only: **{threshold:.2%} forecasted decline**
- 2024-2025 test rows: **{len(test):,}**
- Accuracy: **{tm['Accuracy']:.2%}**
- Balanced Accuracy: **{tm['BalancedAccuracy']:.2%}**
- Precision: **{tm['Precision']:.2%}**
- Recall: **{tm['Recall']:.2%}**
- F1: **{tm['F1']:.2%}**
- ROC-AUC: **{tm['ROC_AUC']:.2%}**
- Majority baseline: **{majority:.2%}**
- All scientific gates passed: **{out['all_gates_passed']}**
''',encoding='utf-8'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
