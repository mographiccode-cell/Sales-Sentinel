from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SEC=ROOT/'data/saudi_v1_5/saudi_sector_daily_panel_v1_5.csv.gz'
MER=ROOT/'data/merchant_v7_1/merchant_feature_panel_v7_1.csv'
OUT=ROOT/'reports/v11_label_reconstruction'; OUT.mkdir(parents=True,exist_ok=True)

s=pd.read_csv(SEC,parse_dates=['TrainingSafeDate'])
m=pd.read_csv(MER,parse_dates=['date']).sort_values('date').reset_index(drop=True)
d=s.groupby('TrainingSafeDate',as_index=True)['sales'].sum().sort_index().to_frame('sales')
full=pd.date_range(d.index.min(),d.index.max(),freq='D')
d=d.reindex(full).fillna(0.0)

# Candidate baselines: inclusive today, previous day only.
d['base_inc']=d.sales.rolling(28,min_periods=28).mean()
d['base_prev']=d.sales.shift(1).rolling(28,min_periods=28).mean()
# Candidate futures: next N days excluding today / including today.
for n in [3,7,14]:
    d[f'fut_excl_{n}']=sum(d.sales.shift(-k) for k in range(1,n+1))
    d[f'fut_incl_{n}']=sum(d.sales.shift(-k) for k in range(0,n))

q=m[['date','baseline28_daily','future7_sales','future_ratio']].merge(d,left_on='date',right_index=True,how='left')

def err(a,b):
    mask=np.isfinite(a)&np.isfinite(b)
    aa=np.asarray(a)[mask]; bb=np.asarray(b)[mask]
    return {'n':int(mask.sum()),'mae':float(np.mean(np.abs(aa-bb))),'rmse':float(np.sqrt(np.mean((aa-bb)**2))),'corr':float(np.corrcoef(aa,bb)[0,1]) if len(aa)>2 else None,'max_abs':float(np.max(np.abs(aa-bb)))}

report={
    'sector_rows':len(s),'daily_rows':len(d),'merchant_rows':len(m),
    'baseline_inc':err(q.baseline28_daily,q.base_inc),
    'baseline_prev':err(q.baseline28_daily,q.base_prev),
    'future_excl_7':err(q.future7_sales,q.fut_excl_7),
    'future_incl_7':err(q.future7_sales,q.fut_incl_7),
}
for b in ['base_inc','base_prev']:
    for f in ['fut_excl_7','fut_incl_7']:
        ratio=q[f]/(7*q[b].replace(0,np.nan))
        report[f'ratio_{b}_{f}']=err(q.future_ratio,ratio)

# Identify exact/best formula and save reconstructed multi-horizon labels with both conventions.
labels=pd.DataFrame({'date':q.date})
for b in ['base_inc','base_prev']:
    labels[b]=q[b]
for n in [3,7,14]:
    labels[f'future{n}_excl']=q[f'fut_excl_{n}']
    labels[f'future{n}_incl']=q[f'fut_incl_{n}']
labels.to_csv(OUT/'candidate_labels.csv',index=False)
(OUT/'diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# V11 Target Reconstruction','',f'- Sector rows: **{len(s)}**',f'- Reconstructed daily dates: **{len(d)}**',f'- Merchant rows compared: **{len(m)}**','']
for k,v in report.items():
    if isinstance(v,dict): lines.append(f'- {k}: MAE={v["mae"]:.6f}, RMSE={v["rmse"]:.6f}, corr={v["corr"]:.8f}, max_abs={v["max_abs"]:.6f}')
(OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print((OUT/'summary.md').read_text())
