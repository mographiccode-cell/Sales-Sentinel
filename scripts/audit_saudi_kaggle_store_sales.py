from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'.saudi_kaggle_audit'; WORK.mkdir(exist_ok=True)
OUT=ROOT/'reports'/'saudi_kaggle_candidate'; OUT.mkdir(parents=True,exist_ok=True)
ZIP=WORK/'sales-in-saudi-arabia.zip'
URL='https://www.kaggle.com/api/v1/datasets/download/shilton123456/sales-in-saudi-arabia'
REPORT=OUT/'audit.json'; SUMMARY=OUT/'audit.md'

def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def norm(s): return re.sub(r'[^a-z0-9]+','',str(s).lower())
def main():
 if not ZIP.exists():
  r=requests.get(URL,timeout=180); r.raise_for_status(); ZIP.write_bytes(r.content)
 ext=WORK/'extracted'; ext.mkdir(exist_ok=True)
 with zipfile.ZipFile(ZIP) as z: z.extractall(ext)
 files=[p for p in ext.rglob('*') if p.is_file()]
 datasets=[]
 for p in files:
  if p.suffix.lower() not in {'.csv','.xlsx','.xls'}: continue
  try:
   df=pd.read_csv(p) if p.suffix.lower()=='.csv' else pd.read_excel(p)
  except Exception as e:
   datasets.append({'file':p.name,'read_error':str(e)}); continue
  cols=list(df.columns); nmap={norm(c):c for c in cols}
  date_col=next((c for k,c in nmap.items() if 'invoicedate' in k or k=='date' or 'saledate' in k or 'transactiondate' in k),None)
  invoice_col=next((c for k,c in nmap.items() if 'invoiceid' in k or 'invoiceno' in k or 'invoice'==k),None)
  sales_col=next((c for k,c in nmap.items() if 'totalsales' in k or 'salesamount' in k or k=='sales' or 'totalamount' in k),None)
  city_col=next((c for k,c in nmap.items() if k=='city' or 'city' in k),None)
  prod_col=next((c for k,c in nmap.items() if 'productname' in k or k=='product'),None)
  cat_col=next((c for k,c in nmap.items() if 'productcategory' in k or k=='category'),None)
  info={'file':p.name,'rows':int(len(df)),'columns':cols,'exact_duplicate_rows':int(df.duplicated().sum()),'null_cells':int(df.isna().sum().sum()),'unique_rows_ratio':float(len(df.drop_duplicates())/max(len(df),1))}
  if date_col:
   dt=pd.to_datetime(df[date_col],errors='coerce'); info.update({'date_column':date_col,'date_parse_success':float(dt.notna().mean()),'date_min':str(dt.min()) if dt.notna().any() else None,'date_max':str(dt.max()) if dt.notna().any() else None,'unique_dates':int(dt.dt.normalize().nunique()),'unique_timestamps':int(dt.nunique()),'date_span_days':int((dt.max()-dt.min()).days) if dt.notna().any() else None,'rows_per_date_median':float(df.assign(_d=dt.dt.normalize()).groupby('_d').size().median()) if dt.notna().any() else None})
  if invoice_col: info.update({'invoice_column':invoice_col,'unique_invoices':int(df[invoice_col].nunique(dropna=True)),'invoice_duplicate_row_ratio':float(1-df[invoice_col].nunique(dropna=True)/max(df[invoice_col].notna().sum(),1))})
  if sales_col:
   s=pd.to_numeric(df[sales_col],errors='coerce'); info.update({'sales_column':sales_col,'sales_numeric_rate':float(s.notna().mean()),'sales_min':float(s.min()) if s.notna().any() else None,'sales_median':float(s.median()) if s.notna().any() else None,'sales_mean':float(s.mean()) if s.notna().any() else None,'sales_max':float(s.max()) if s.notna().any() else None,'sales_unique_values':int(s.nunique()),'sales_round_integer_rate':float(np.isclose(s.dropna()%1,0).mean()) if s.notna().any() else None})
  for label,col in [('city',city_col),('product',prod_col),('category',cat_col)]:
   if col: info[f'{label}_column']=col; info[f'unique_{label}s']=int(df[col].nunique(dropna=True)); info[f'{label}_top10']=df[col].value_counts(dropna=False).head(10).to_dict()
  # Heuristics only; not proof of synthetic generation.
  flags=[]
  if len(df)<5000: flags.append('small_row_count')
  if info.get('unique_dates',0)<180: flags.append('short_or_sparse_date_history')
  if info.get('date_span_days',0)<365: flags.append('less_than_one_year_history')
  if info.get('unique_invoices')==len(df): flags.append('exactly_one_row_per_invoice')
  if info.get('exact_duplicate_rows',0)==0: flags.append('perfect_no_exact_duplicates_not_proof_but_review')
  if info.get('null_cells',0)==0: flags.append('perfect_no_nulls_not_proof_but_review')
  if info.get('unique_products',9999)<=20: flags.append('very_small_product_catalog')
  info['synthetic_risk_heuristics']=flags
  datasets.append(info)
 result={'source_url':URL,'download_sha256':sha(ZIP),'files':[str(p.relative_to(ext)) for p in files],'datasets':datasets,'decision':None}
 candidates=[x for x in datasets if not x.get('read_error')]
 if not candidates: decision='REJECT_NO_READABLE_TABULAR_DATA'
 else:
  best=max(candidates,key=lambda x:x.get('rows',0)); reasons=[]
  if best.get('rows',0)<10000: reasons.append('fewer than 10k rows')
  if best.get('unique_dates',0)<365: reasons.append('fewer than 365 distinct dates')
  if best.get('date_span_days',0)<365: reasons.append('history shorter than one year')
  if not best.get('sales_column'): reasons.append('no clear total-sales column')
  if not best.get('invoice_column'): reasons.append('no invoice identifier')
  decision='REJECT_FOR_SALES_SENTINEL' if reasons else 'CANDIDATE_NEEDS_PROVENANCE_VERIFICATION'
  result['decision']=decision; result['decision_reasons']=reasons; result['best_file']=best.get('file')
 REPORT.write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
 lines=['# Saudi Kaggle Store Sales Audit','',f"- Decision: **{result['decision']}**",f"- Download SHA-256: `{result['download_sha256']}`"]
 for x in candidates:
  lines += ['',f"## {x['file']}",f"- Rows: **{x.get('rows',0):,}**",f"- Date: **{x.get('date_min')} → {x.get('date_max')}**",f"- Unique dates: **{x.get('unique_dates')}**",f"- Unique invoices: **{x.get('unique_invoices')}**",f"- Duplicate rows: **{x.get('exact_duplicate_rows')}**",f"- Null cells: **{x.get('null_cells')}**",f"- Synthetic-risk heuristics: **{', '.join(x.get('synthetic_risk_heuristics',[])) or 'none'}**"]
 if result.get('decision_reasons'): lines += ['', '- Rejection reasons: ' + '; '.join(result['decision_reasons'])]
 lines += ['', 'Important: absence/presence of heuristic flags does not prove provenance. A dataset must have a credible source statement before it can replace UCI as observed Saudi merchant microdata.']
 SUMMARY.write_text('\n'.join(lines),encoding='utf-8'); print(json.dumps(result,indent=2,default=str))
if __name__=='__main__': main()
