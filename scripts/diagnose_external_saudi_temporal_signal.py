from __future__ import annotations

import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
RAW_DIR=Path(os.environ.get('SAUDI_STORE_DIR','/tmp/saudi_store_sales'))
OUT=ROOT/'reports/external_saudi_temporal_signal'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'diagnostics.json'; SUMMARY=OUT/'summary.md'; MONTHLY=OUT/'monthly_target_prevalence.csv'

def load():
    files=list(RAW_DIR.rglob('*.xlsx'))+list(RAW_DIR.rglob('*.csv'))
    if not files: raise FileNotFoundError(RAW_DIR)
    p=sorted(files)[0]
    d=pd.read_excel(p) if p.suffix.lower()=='.xlsx' else pd.read_csv(p,low_memory=False)
    d['Invoice Date']=pd.to_datetime(d['Invoice Date'],errors='coerce'); d['Total Sales']=pd.to_numeric(d['Total Sales'],errors='coerce')
    return d.dropna(subset=['Invoice Date','Total Sales']).copy()

def entropy_norm(s):
    p=s.value_counts(normalize=True,dropna=False).to_numpy(float)
    h=-np.sum(p*np.log(p+1e-12)); return float(h/np.log(len(p))) if len(p)>1 else 0.0

def variance_explained(values,groups):
    v=np.asarray(values,float); g=np.asarray(groups); mu=np.nanmean(v); total=np.nansum((v-mu)**2)
    between=0.0
    for k in pd.unique(g):
        x=v[g==k]
        if len(x): between += len(x)*(np.nanmean(x)-mu)**2
    return float(between/total) if total>0 else 0.0

def main():
    tx=load(); tx['date']=tx['Invoice Date'].dt.normalize()
    daily=tx.groupby('date').agg(sales=('Total Sales','sum'),rows=('Invoice ID','size'),invoices=('Invoice ID','nunique')).sort_index()
    daily=daily.reindex(pd.date_range(daily.index.min(),daily.index.max(),freq='D')).fillna(0)
    s=daily.sales.astype(float)
    for lag in [1,2,3,7,14,28,56]: daily[f'ac_lag{lag}']=s.shift(lag)
    ac={str(lag):float(s.autocorr(lag)) for lag in [1,2,3,7,14,28,56]}
    daily['mean7']=s.rolling(7,min_periods=7).mean(); daily['mean28']=s.rolling(28,min_periods=28).mean(); daily['baseline28']=daily.mean28
    daily['future7']=sum(s.shift(-k) for k in range(1,8)); daily['future_ratio']=daily.future7/(7*daily.baseline28.replace(0,np.nan)); daily['target']=(daily.future_ratio<.85).astype(float); daily.loc[daily.future_ratio.isna(),'target']=np.nan
    daily['recent_ratio']=daily.mean7/daily.mean28.replace(0,np.nan); daily['risk_simple']=-daily.recent_ratio
    valid=daily.target.notna()&daily.risk_simple.notna(); y=daily.loc[valid,'target'].astype(int); score=daily.loc[valid,'risk_simple']
    simple={'roc_auc':float(roc_auc_score(y,score)),'pr_auc':float(average_precision_score(y,score)),'positives':int(y.sum()),'rows':int(len(y))}
    per_year={}
    for yr,g in daily.loc[valid].groupby(daily.loc[valid].index.year):
        yy=g.target.astype(int); ss=g.risk_simple
        per_year[str(yr)]={'rows':len(g),'positive_rate':float(yy.mean()),'sales_mean':float(g.sales.mean()),'sales_std':float(g.sales.std()),'sales_cv':float(g.sales.std()/g.sales.mean()),'simple_auc':float(roc_auc_score(yy,ss)) if yy.nunique()==2 else None,'simple_pr':float(average_precision_score(yy,ss)) if yy.nunique()==2 else None}
    tmp=daily.loc[valid].copy(); tmp['month']=tmp.index.to_period('M').astype(str); monthly=tmp.groupby('month').agg(rows=('target','size'),positives=('target','sum'),positive_rate=('target','mean'),sales_mean=('sales','mean'),sales_std=('sales','std')).reset_index(); monthly.to_csv(MONTHLY,index=False)
    weekday_effect=variance_explained(s.to_numpy(),daily.index.dayofweek); month_effect=variance_explained(s.to_numpy(),daily.index.month); year_effect=variance_explained(s.to_numpy(),daily.index.year)
    row_ac={str(lag):float(daily.rows.autocorr(lag)) for lag in [1,7,14,28]}
    cats={}
    for c in ['Customer Name','Employee Name','Manager Name','Product Name','Product Category','City','Channel','Customer Type','Customer Satisfaction','Invoice ID']:
        if c in tx.columns: cats[c]={'unique':int(tx[c].nunique(dropna=False)),'normalized_entropy':entropy_norm(tx[c]),'top_share':float(tx[c].value_counts(normalize=True,dropna=False).iloc[0])}
    # Counts by year and weekday assess whether transactions are spread almost uniformly over calendar.
    tx['year']=tx['date'].dt.year; tx['dow']=tx['date'].dt.dayofweek
    year_counts={str(int(k)):int(v) for k,v in tx.year.value_counts().sort_index().items()}; dow_counts={str(int(k)):int(v) for k,v in tx.dow.value_counts().sort_index().items()}
    report={'rows':len(tx),'days':len(daily),'daily_sales_autocorrelation':ac,'daily_row_count_autocorrelation':row_ac,'variance_explained_by_weekday':weekday_effect,'variance_explained_by_month':month_effect,'variance_explained_by_year':year_effect,'simple_recent7_vs28_baseline':simple,'per_year':per_year,'categorical_cardinality':cats,'transaction_counts_by_year':year_counts,'transaction_counts_by_weekday':dow_counts,'daily_rows':{'mean':float(daily.rows.mean()),'std':float(daily.rows.std()),'cv':float(daily.rows.std()/daily.rows.mean()),'min':int(daily.rows.min()),'max':int(daily.rows.max())}}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    lines=['# External Saudi Dataset Temporal-Signal Diagnostics','',f'- Transactions / days: **{len(tx):,} / {len(daily):,}**',f'- Daily sales autocorrelation lag1 / lag7 / lag28: **{ac["1"]:.3f} / {ac["7"]:.3f} / {ac["28"]:.3f}**',f'- Variance explained by weekday / month / year: **{weekday_effect:.2%} / {month_effect:.2%} / {year_effect:.2%}**',f'- Daily row-count CV: **{report["daily_rows"]["cv"]:.2%}**',f'- Simple recent-7-vs-28 risk AUC / PR-AUC: **{simple["roc_auc"]:.2%} / {simple["pr_auc"]:.2%}**','', '## Per year']
    for yr,z in per_year.items(): lines.append(f'- {yr}: rows={z["rows"]}, decline prevalence={z["positive_rate"]:.2%}, sales CV={z["sales_cv"]:.2%}, simple AUC={z["simple_auc"]:.2%}' if z['simple_auc'] is not None else f'- {yr}: {z}')
    lines += ['', '## Cardinality / repetition']
    for c,z in cats.items(): lines.append(f'- {c}: unique={z["unique"]:,}, top share={z["top_share"]:.2%}, normalized entropy={z["normalized_entropy"]:.3f}')
    lines += ['',f'- Transactions by year: **{year_counts}**',f'- Transactions by weekday: **{dow_counts}**','', 'Interpretation rule: very low temporal autocorrelation and negligible calendar variance imply weak forecasting signal, regardless of whether the table is geographically labeled Saudi. This diagnostic does not by itself prove synthetic generation.']
    SUMMARY.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(SUMMARY.read_text())
if __name__=='__main__': main()
