from __future__ import annotations

import hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=Path(os.environ.get('SAUDI_STORE_DIR','/tmp/saudi_store_sales'))
OUT=ROOT/'reports/external_saudi_store_sales_inspection'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'inspection_report.json'; SUMMARY=OUT/'inspection_summary.md'

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def load(p):
    if p.suffix.lower()=='.csv': return pd.read_csv(p,low_memory=False)
    if p.suffix.lower() in {'.xlsx','.xls'}: return pd.read_excel(p)
    if p.suffix.lower()=='.parquet': return pd.read_parquet(p)
    raise ValueError(p.suffix)

def main():
    files=sorted([p for p in SRC.rglob('*') if p.is_file()]); inspected=[]
    for p in files:
        item={'name':p.name,'size_bytes':p.stat().st_size,'sha256':sha256(p)}
        if p.suffix.lower() not in {'.csv','.xlsx','.xls','.parquet'}:
            item['status']='non_tabular'; inspected.append(item); continue
        try:
            d=load(p); item.update(status='ok',rows=int(len(d)),columns=int(len(d.columns)),column_names=[str(c) for c in d.columns],duplicate_rows=int(d.duplicated().sum()),missing_cells=int(d.isna().sum().sum()))
            date_candidates=[c for c in d.columns if any(k in str(c).lower() for k in ['date','time','timestamp'])]
            ranges={}
            for c in date_candidates:
                q=pd.to_datetime(d[c],errors='coerce')
                if q.notna().sum()>=max(3,int(.2*len(d))): ranges[str(c)]={'min':str(q.min()),'max':str(q.max()),'unique':int(q.nunique()),'parsed':int(q.notna().sum())}
            item['date_ranges']=ranges
            sem={}
            for k in ['invoice','customer','product','category','city','channel','sales','total','amount','quantity','branch','store']:
                sem[k]=[str(c) for c in d.columns if k in str(c).lower()]
            item['semantic_columns']=sem
            num={}
            for c in d.select_dtypes(include=[np.number]).columns:
                x=pd.to_numeric(d[c],errors='coerce')
                if x.notna().any(): num[str(c)]={'min':float(x.min()),'max':float(x.max()),'mean':float(x.mean()),'std':float(x.std()),'unique':int(x.nunique())}
            item['numeric_summary']=num
            # temporal granularity and repeated rows by date/city to assess suitability for forecasting
            item['sample_rows']=d.head(8).astype(str).to_dict(orient='records')
            if ranges:
                dc=next(iter(ranges)); q=pd.to_datetime(d[dc],errors='coerce'); item['daily_row_stats']={'days':int(q.dt.date.nunique()),'median_rows_per_day':float(q.groupby(q.dt.date).size().median()),'max_rows_per_day':int(q.groupby(q.dt.date).size().max())}
        except Exception as e: item.update(status='error',error=repr(e))
        inspected.append(item)
    usable=[x for x in inspected if x.get('status')=='ok']; report={'source':'Kaggle shilton123456/sales-in-saudi-arabia','files':inspected,'usable_files':len(usable),'total_rows':sum(x.get('rows',0) for x in usable)}; REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Saudi Store Sales Dataset Inspection','',f'- Source: **Kaggle shilton123456/sales-in-saudi-arabia**',f'- Files: **{len(files)}**',f'- Usable tabular files: **{len(usable)}**',f'- Total rows: **{report["total_rows"]:,}**','']
    for x in usable: lines += [f'## {x["name"]}',f'- Rows / columns: **{x["rows"]:,} / {x["columns"]}**',f'- SHA-256: `{x["sha256"]}`',f'- Duplicates / missing cells: **{x["duplicate_rows"]:,} / {x["missing_cells"]:,}**',f'- Columns: {x["column_names"]}',f'- Date ranges: {x.get("date_ranges",{})}',f'- Daily row stats: {x.get("daily_row_stats",{})}',f'- Semantic columns: {x.get("semantic_columns",{})}','']
    SUMMARY.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(SUMMARY.read_text())
if __name__=='__main__': main()
