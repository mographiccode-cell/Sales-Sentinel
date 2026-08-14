from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
HIST=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2020_2025.csv'
FCST=ROOT/'data'/'sama_pos'/'sama_sector_walkforward_forecasts_2023_2025.csv'
OUT=ROOT/'reports'/'sama_sector_v1_8'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'sama_sector_decline_v1_8.json'; SUMMARY=OUT/'sama_sector_decline_v1_8.md'
DECLINE=.20


def metrics(y,score):
    pred=(np.asarray(score)>=0).astype(int); y=np.asarray(y).astype(int)
    return {
        'Accuracy':float(accuracy_score(y,pred)),
        'BalancedAccuracy':float(balanced_accuracy_score(y,pred)),
        'Precision':float(precision_score(y,pred,zero_division=0)),
        'Recall':float(recall_score(y,pred,zero_division=0)),
        'F1':float(f1_score(y,pred,zero_division=0)),
        'ROC_AUC':float(roc_auc_score(y,score)),
        'ConfusionMatrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
    }

def main():
    h=pd.read_csv(HIST,parse_dates=['week_start']).sort_values(['sector','week_start'])
    f=pd.read_csv(FCST,parse_dates=['origin_week_start','forecast_h1_week_start'])
    # Baseline is strictly known at prediction origin: mean of origin week and previous 3 completed sector weeks.
    h['baseline4']=h.groupby('sector').value_thousand_sar.transform(lambda s:s.rolling(4,min_periods=4).mean())
    base=h[['week_start','sector','baseline4','value_thousand_sar']].rename(columns={'week_start':'origin_week_start','value_thousand_sar':'origin_value'})
    d=f.merge(base,on=['origin_week_start','sector'],how='left',validate='many_to_one')
    d=d.dropna(subset=['baseline4','predicted_value_h1','actual_value_h1']).copy()
    d['target']=(d.actual_value_h1 < (1-DECLINE)*d.baseline4).astype(int)
    # score > 0 means forecasted next week is below the fixed 20% decline boundary.
    d['risk_score']=((1-DECLINE)*d.baseline4-d.predicted_value_h1)/d.baseline4
    d['predicted_decline']=(d.risk_score>=0).astype(int)

    periods={
        '2023':d[(d.origin_week_start>='2023-01-01')&(d.origin_week_start<'2024-01-01')],
        '2024':d[(d.origin_week_start>='2024-01-01')&(d.origin_week_start<'2025-01-01')],
        '2025':d[d.origin_week_start>='2025-01-01'],
        '2024_2025':d[d.origin_week_start>='2024-01-01'],
    }
    out={'version':'SAMA-SECTOR-DECLINE-1.8','source':'Official SAMA national sector weekly POS value; predictions are leakage-safe walk-forward forecasts','target':'next official SAMA sector week is >=20% below trailing 4 completed weeks mean','prediction_rule':'predicted h1 sector value is >=20% below the same known trailing-4-week baseline','fixed_rule_no_test_threshold_tuning':True,'periods':{}}
    for name,q in periods.items():
        if q.empty or q.target.nunique()<2:
            out['periods'][name]={'rows':int(len(q)),'positive_rate':float(q.target.mean()) if len(q) else None,'metrics':None}; continue
        out['periods'][name]={'rows':int(len(q)),'positive_rate':float(q.target.mean()),'metrics':metrics(q.target,q.risk_score)}
    q=periods['2024_2025']; m=out['periods']['2024_2025']['metrics']; majority=max(float(q.target.mean()),1-float(q.target.mean()))
    gates={'accuracy_at_least_90pct':m['Accuracy']>=.90,'balanced_accuracy_at_least_80pct':m['BalancedAccuracy']>=.80,'recall_at_least_70pct':m['Recall']>=.70,'f1_at_least_65pct':m['F1']>=.65,'roc_auc_at_least_85pct':m['ROC_AUC']>=.85,'beats_majority':m['Accuracy']>majority}
    out['combined_2024_2025_majority_accuracy']=majority; out['acceptance_gates']=gates; out['all_gates_passed']=bool(all(gates.values()))
    REPORT.write_text(json.dumps(out,indent=2),encoding='utf-8')
    SUMMARY.write_text(f'''# SAMA Sector Decline v1.8

- Evaluation: official SAMA sector-week values only
- Target: next week decline >=20% vs trailing 4-week known baseline
- 2024-2025 rows: **{len(q):,}**
- Positive rate: **{q.target.mean():.2%}**
- Accuracy: **{m['Accuracy']:.2%}**
- Balanced Accuracy: **{m['BalancedAccuracy']:.2%}**
- Precision: **{m['Precision']:.2%}**
- Recall: **{m['Recall']:.2%}**
- F1: **{m['F1']:.2%}**
- ROC-AUC: **{m['ROC_AUC']:.2%}**
- Majority baseline: **{majority:.2%}**
- 90% accuracy gate: **{gates['accuracy_at_least_90pct']}**
- All scientific gates passed: **{out['all_gates_passed']}**
''',encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
