from __future__ import annotations

import hashlib, json, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
SRC=Path(os.environ.get('REDSEA_FILE','/tmp/redsea_mendeley/RedSea_Data_Cleaned.xlsx'))
OUT=ROOT/'reports/redsea_temporal_audit'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'audit.json'; SUMMARY=OUT/'summary.md'; DAILY=OUT/'daily_series.csv'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def safe_auc(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float)
    if len(np.unique(y))<2:return None,None
    return float(roc_auc_score(y,s)),float(average_precision_score(y,s))

def main():
    raw=pd.read_excel(SRC)
    d=raw.copy(); d['TRX DATE']=pd.to_datetime(d['TRX DATE'],errors='coerce')
    nums=['QUANTITY','Unit Price','Discount Amount','Discount Amount(%)','Net Amount','Vat Amount','TOTAL AMOUNT']
    for c in nums:d[c]=pd.to_numeric(d[c],errors='coerce')
    missing={c:int(d[c].isna().sum()) for c in d.columns if d[c].isna().any()}
    type_counts={str(k):int(v) for k,v in d['Type'].value_counts(dropna=False).items()}
    channel_counts={str(k):int(v) for k,v in d['SALES CHANNEL'].value_counts(dropna=False).items()}
    org_counts={str(k):int(v) for k,v in d['ORG'].value_counts(dropna=False).items()}
    audit={
        'sha256':sha(SRC),'raw_rows':len(d),'columns':len(d.columns),'exact_duplicate_rows':int(d.duplicated().sum()),
        'unique_trx_numbers':int(d['TRX NUMBER'].nunique()),'unique_customers':int(d['CUSTOMER NUMBER'].nunique()),'unique_items':int(d['ITEM CODE'].nunique()),
        'unique_families':int(d['FAMILY'].nunique()),'unique_classes':int(d['CLASS'].nunique()),'date_min':str(d['TRX DATE'].min().date()),'date_max':str(d['TRX DATE'].max().date()),'active_dates':int(d['TRX DATE'].dt.normalize().nunique()),
        'missing_by_column':missing,'type_counts':type_counts,'channel_counts':channel_counts,'org_counts':org_counts,
        'negative_quantity_rows':int((d.QUANTITY<0).sum()),'zero_quantity_rows':int((d.QUANTITY==0).sum()),'negative_net_amount_rows':int((d['Net Amount']<0).sum()),'negative_total_amount_rows':int((d['TOTAL AMOUNT']<0).sum()),
        'discount_nonzero_rows':int((d['Discount Amount'].fillna(0)!=0).sum()),
        'quantity_sum':float(d.QUANTITY.sum()),'net_amount_sum_raw':float(d['Net Amount'].sum()),'total_amount_sum_raw':float(d['TOTAL AMOUNT'].sum()),
    }
    # Exact duplicates are removed only for the canonical audit series; report their monetary effect.
    dup=d[d.duplicated(keep='first')]
    audit['duplicate_effect']={'rows':len(dup),'net_amount':float(dup['Net Amount'].sum()),'total_amount':float(dup['TOTAL AMOUNT'].sum()),'quantity':float(dup.QUANTITY.sum())}
    clean=d.drop_duplicates().copy()
    audit['clean_rows']=len(clean)

    # TRX NUMBER is invoice/transaction identifier; multiple line items per transaction are preserved.
    clean['date']=clean['TRX DATE'].dt.normalize()
    grouped=clean.groupby('date').agg(
        net_sales=('Net Amount','sum'),gross_total=('TOTAL AMOUNT','sum'),vat=('Vat Amount','sum'),discount=('Discount Amount','sum'),quantity=('QUANTITY','sum'),
        line_items=('TRX NUMBER','size'),transactions=('TRX NUMBER','nunique'),customers=('CUSTOMER NUMBER','nunique'),products=('ITEM CODE','nunique'),families=('FAMILY','nunique'),classes=('CLASS','nunique')
    ).sort_index()
    ret=clean.assign(is_return=(clean.QUANTITY<0)|(clean['Net Amount']<0)).groupby('date')['is_return'].mean().rename('return_line_share')
    grouped=grouped.join(ret)
    full=pd.date_range(grouped.index.min(),grouped.index.max(),freq='D'); grouped=grouped.reindex(full)
    count_cols=['net_sales','gross_total','vat','discount','quantity','line_items','transactions','customers','products','families','classes','return_line_share']
    grouped[count_cols]=grouped[count_cols].fillna(0.0); grouped.index.name='date'
    audit['calendar_days']=len(grouped); audit['zero_transaction_days']=int((grouped.transactions==0).sum())
    audit['daily']={'net_sales_mean':float(grouped.net_sales.mean()),'net_sales_std':float(grouped.net_sales.std()),'net_sales_cv':float(grouped.net_sales.std()/grouped.net_sales.mean()),'transactions_mean':float(grouped.transactions.mean()),'transactions_std':float(grouped.transactions.std()),'transactions_cv':float(grouped.transactions.std()/grouped.transactions.mean())}
    audit['autocorrelation']={str(l):float(grouped.net_sales.autocorr(l)) for l in [1,2,3,7,14,28]}

    # Same frozen Sales Sentinel target: next 7 days vs trailing 28-day mean including prediction day.
    grouped['baseline28_daily']=grouped.net_sales.rolling(28,min_periods=28).mean()
    grouped['future7_sales']=sum(grouped.net_sales.shift(-k) for k in range(1,8))
    grouped['future_ratio']=grouped.future7_sales/(7*grouped.baseline28_daily.replace(0,np.nan))
    grouped['target']=np.where(grouped.future_ratio.notna(),(grouped.future_ratio<.85).astype(float),np.nan)
    grouped['mean7']=grouped.net_sales.rolling(7,min_periods=7).mean(); grouped['recent7_28']=grouped.mean7/grouped.baseline28_daily.replace(0,np.nan); grouped['simple_risk']=-grouped.recent7_28
    valid=grouped.target.notna()&grouped.simple_risk.notna(); y=grouped.loc[valid,'target'].astype(int); score=grouped.loc[valid,'simple_risk']; auc,pr=safe_auc(y,score)
    audit['target']={'usable_rows':int(valid.sum()),'positives':int(y.sum()),'positive_rate':float(y.mean()),'first_usable_date':str(grouped.loc[valid].index.min().date()),'last_usable_date':str(grouped.loc[valid].index.max().date()),'simple_recent7_28_auc':auc,'simple_recent7_28_pr_auc':pr}
    # Monthly/event prevalence for diagnostic only.
    tmp=grouped.loc[valid].copy(); tmp['month']=tmp.index.to_period('M').astype(str)
    audit['monthly_target']={str(k):{'rows':int(len(g)),'positives':int(g.target.sum()),'rate':float(g.target.mean())} for k,g in tmp.groupby('month')}
    grouped.reset_index().to_csv(DAILY,index=False)
    REPORT.write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    t=audit['target']
    lines=['# Redsea Real Saudi Merchant — Temporal & Accounting Audit','',f'- SHA-256: `{audit["sha256"]}`',f'- Raw / clean rows: **{audit["raw_rows"]:,} / {audit["clean_rows"]:,}**',f'- Exact duplicates: **{audit["exact_duplicate_rows"]}**',f'- Unique transactions / customers / items: **{audit["unique_trx_numbers"]:,} / {audit["unique_customers"]:,} / {audit["unique_items"]:,}**',f'- Active / calendar dates: **{audit["active_dates"]} / {audit["calendar_days"]}**',f'- Zero-transaction days: **{audit["zero_transaction_days"]}**',f'- Missing cells by column: **{missing}**',f'- Type counts: **{type_counts}**',f'- Negative quantity / net-amount rows: **{audit["negative_quantity_rows"]} / {audit["negative_net_amount_rows"]}**',f'- Duplicate monetary effect (Net Amount): **SAR {audit["duplicate_effect"]["net_amount"]:,.2f}**','', '## Temporal signal',f'- Net sales daily CV: **{audit["daily"]["net_sales_cv"]:.2%}**',f'- Autocorrelation lag1 / lag7 / lag28: **{audit["autocorrelation"]["1"]:.3f} / {audit["autocorrelation"]["7"]:.3f} / {audit["autocorrelation"]["28"]:.3f}**',f'- Frozen 7-day target usable rows: **{t["usable_rows"]}**',f'- Decline positives / prevalence: **{t["positives"]} / {t["positive_rate"]:.2%}**',f'- Target dates: **{t["first_usable_date"]} → {t["last_usable_date"]}**',f'- Simple recent7-vs28 AUC / PR-AUC: **{t["simple_recent7_28_auc"]:.2%} / {t["simple_recent7_28_pr_auc"]:.2%}**','',f'- Monthly target: **{audit["monthly_target"]}**','', 'Scientific handling: exact duplicate rows are removed; negative quantities/amounts are retained as observed returns/adjustments unless later transaction-type evidence justifies a different treatment. Net Amount (before VAT) is used for the Sales Sentinel net-sales target.']
    SUMMARY.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(SUMMARY.read_text())
if __name__=='__main__': main()
