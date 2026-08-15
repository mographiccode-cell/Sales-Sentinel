from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

VERSION='SALES-SENTINEL-REAL-MERCHANT-DATA-CONTRACT-4.0'
REQUIRED={'date','invoice_id','product_id','quantity','unit_price','category'}
RECOMMENDED={'customer_id','branch_id','discount','promotion_id','return_flag','inventory_on_hand','stockout_flag','payment_type'}


def validate(path:Path)->dict:
    if not path.exists():
        return {'version':VERSION,'valid':False,'reason':f'FILE_NOT_FOUND:{path}'}
    if path.suffix.lower() in {'.xlsx','.xls'}:
        d=pd.read_excel(path)
    else:
        d=pd.read_csv(path)
    cols={str(c).strip().lower() for c in d.columns}
    missing=sorted(REQUIRED-cols)
    recommended_missing=sorted(RECOMMENDED-cols)
    out={'version':VERSION,'rows':int(len(d)),'columns':sorted(cols),'missing_required':missing,'missing_recommended':recommended_missing}
    if missing:
        out.update({'valid':False,'reason':'MISSING_REQUIRED_COLUMNS'});return out
    x=d.copy();x.columns=[str(c).strip().lower() for c in x.columns];x['date']=pd.to_datetime(x.date,errors='coerce')
    bad_date=float(x.date.isna().mean());start=x.date.min();end=x.date.max();days=int((end-start).days+1) if pd.notna(start) and pd.notna(end) else 0
    dup=float(x.duplicated().mean());invoice_null=float(x.invoice_id.isna().mean());product_null=float(x.product_id.isna().mean())
    longitudinal_invoice=bool(x.groupby('invoice_id').size().max()>1) if len(x) else False
    product_repeat=bool(x.groupby('product_id').size().max()>1) if len(x) else False
    customer_repeat=None
    if 'customer_id' in x: customer_repeat=bool(x.dropna(subset=['customer_id']).groupby('customer_id').size().max()>1) if x.customer_id.notna().any() else False
    gates={'date_parse_failure_le_0_1pct':bad_date<=.001,'date_span_ge_180_days':days>=180,'rows_ge_10000':len(x)>=10000,'duplicate_rate_le_2pct':dup<=.02,'invoice_id_present':invoice_null<=.001,'product_id_present':product_null<=.001,'invoice_identity_longitudinal':longitudinal_invoice,'product_identity_longitudinal':product_repeat}
    out.update({'date_start':None if pd.isna(start) else str(start.date()),'date_end':None if pd.isna(end) else str(end.date()),'date_span_days':days,'date_parse_failure_rate':bad_date,'duplicate_rate':dup,'customer_identity_repeats':customer_repeat,'gates':gates,'valid':bool(all(gates.values())),'scientific_note':'Anonymized IDs are acceptable, but invoice/product/customer identities must remain stable; randomly reassigning IDs per row destroys longitudinal features.'})
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('file');args=p.parse_args();print(json.dumps(validate(Path(args.file)),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
