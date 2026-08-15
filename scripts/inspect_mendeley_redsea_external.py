from __future__ import annotations

import hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=Path(os.environ.get('REDSEA_DIR','/tmp/redsea_mendeley'))
OUT=ROOT/'reports/external_redsea_mendeley_inspection'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'inspection_report.json'; SUMMARY=OUT/'inspection_summary.md'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
    return h.hexdigest()

def read(p):
    ext=p.suffix.lower()
    if ext=='.csv': return pd.read_csv(p,low_memory=False)
    if ext in {'.xlsx','.xls'}: return pd.read_excel(p)
    if ext=='.sav':
        try: return pd.read_spss(p)
        except Exception: raise
    if ext=='.parquet': return pd.read_parquet(p)
    raise ValueError(ext)

def main():
    files=sorted([p for p in SRC.rglob('*') if p.is_file()]); rows=[]
    for p in files:
        x={'name':p.name,'size_bytes':p.stat().st_size,'sha256':sha(p)}
        if p.suffix.lower() not in {'.csv','.xlsx','.xls','.sav','.parquet'}:
            x['status']='non_tabular'; rows.append(x); continue
        try:
            d=read(p); x.update(status='ok',rows=int(len(d)),columns=int(len(d.columns)),column_names=[str(c) for c in d.columns],duplicates=int(d.duplicated().sum()),missing_cells=int(d.isna().sum().sum()))
            date_candidates=[c for c in d.columns if any(k in str(c).lower() for k in ['date','time','day','month','year','invoice'])]
            ranges={}
            for c in date_candidates:
                q=pd.to_datetime(d[c],errors='coerce')
                if q.notna().sum()>=max(3,int(.15*len(d))): ranges[str(c)]={'min':str(q.min()),'max':str(q.max()),'unique':int(q.nunique()),'parsed':int(q.notna().sum())}
            x['date_ranges']=ranges
            sem={}
            for k in ['transaction','invoice','customer','product','category','sales','price','amount','quantity','region','branch','store','showroom','city','date','revenue','total']:
                sem[k]=[str(c) for c in d.columns if k in str(c).lower()]
            x['semantic_columns']=sem
            x['sample_rows']=d.head(5).astype(str).to_dict(orient='records')
            nums={}
            for c in d.select_dtypes(include=[np.number]).columns:
                q=pd.to_numeric(d[c],errors='coerce')
                if q.notna().any(): nums[str(c)]={'min':float(q.min()),'max':float(q.max()),'mean':float(q.mean()),'std':float(q.std()),'unique':int(q.nunique())}
            x['numeric_summary']=nums
        except Exception as e: x.update(status='error',error=repr(e))
        rows.append(x)
    usable=[x for x in rows if x.get('status')=='ok']; report={'source':'Mendeley Data Redsea Dataset','doi':'10.17632/9c87bd42ct.1','files_downloaded':len(files),'usable_tabular_files':len(usable),'files':rows}; REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Mendeley Redsea Dataset Inspection','', '- DOI: **10.17632/9c87bd42ct.1**',f'- Files downloaded: **{len(files)}**',f'- Usable tabular files: **{len(usable)}**','']
    for x in usable: lines += [f'## {x["name"]}',f'- Rows / columns: **{x["rows"]:,} / {x["columns"]}**',f'- SHA-256: `{x["sha256"]}`',f'- Duplicates / missing: **{x["duplicates"]:,} / {x["missing_cells"]:,}**',f'- Columns: {x["column_names"]}',f'- Date ranges: {x["date_ranges"]}',f'- Semantic columns: {x["semantic_columns"]}','']
    SUMMARY.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print(SUMMARY.read_text())
if __name__=='__main__': main()
