from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests

import fetch_sama_pos_calibration as src

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data'/'sama_pos'; REPORT=ROOT/'reports'/'sama_city_v2_1'
OUT.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
CITY=OUT/'sama_city_weekly_value_count_2020_2025.csv'
AUDIT=REPORT/'history_audit.json'


def main():
    r=requests.get(src.EXPORT_URL,timeout=180,headers={'User-Agent':'Sales-Sentinel-Academic/2.1'})
    r.raise_for_status(); raw=src.read_export(r.content); cols=list(raw.columns)
    date_col=src.resolve_column(cols,['starting_date','starting date'])
    indicator_col=src.resolve_column(cols,['number_value_change_transactions','indicator'])
    sector_col=src.resolve_column(cols,['sectors','sector'])
    city_col=src.resolve_column(cols,['city'])
    value_col=src.resolve_column(cols,['value'])
    d=raw[[date_col,indicator_col,sector_col,city_col,value_col]].copy(); d.columns=['week_start','indicator','sector','city','value']
    d.week_start=pd.to_datetime(d.week_start,errors='coerce'); d.value=pd.to_numeric(d.value,errors='coerce'); d=d.dropna()
    d=d[d.week_start.between('2020-01-01','2025-07-06')].copy()
    ind=d.indicator.astype(str).str.lower(); sector=d.sector.astype(str).str.strip().str.lower(); city=d.city.astype(str).str.strip()
    value_mask=ind.str.contains('value')&ind.str.contains('transaction')&~ind.str.contains('change')
    count_mask=ind.str.contains('number')&ind.str.contains('transaction')&~ind.str.contains('change')
    # City totals are rows with Sector=Total and City != Total. Their meaning is independent of sector taxonomy.
    base=d[sector.eq('total') & ~city.str.lower().eq('total')].copy()
    v=base[value_mask.loc[base.index]][['week_start','city','value']].rename(columns={'value':'value_thousand_sar'})
    c=base[count_mask.loc[base.index]][['week_start','city','value']].rename(columns={'value':'transaction_count_thousand'})
    panel=v.merge(c,on=['week_start','city'],how='inner',validate='one_to_one').sort_values(['city','week_start']).reset_index(drop=True)
    panel['week_end']=panel.week_start+pd.Timedelta(days=6)
    panel=panel[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']]
    city_stats=panel.groupby('city').agg(rows=('week_start','size'),start=('week_start','min'),end=('week_start','max')).reset_index()
    gaps={name:int(q.week_start.sort_values().diff().dt.days.dropna().max()) for name,q in panel.groupby('city')}
    weeks=panel.groupby('week_start').city.nunique()
    checks={
        'at_least_8_cities':panel.city.nunique()>=8,
        'at_least_200_weeks':panel.week_start.nunique()>=200,
        'at_least_1800_rows':len(panel)>=1800,
        'no_duplicate_city_weeks':not panel.duplicated(['week_start','city']).any(),
        'all_values_positive':bool((panel.value_thousand_sar>0).all()),
        'all_counts_positive':bool((panel.transaction_count_thousand>0).all()),
        'weekly_continuity_all_cities':max(gaps.values())<=7,
        'constant_city_coverage':weeks.nunique()==1,
    }
    audit={
        'version':'SAMA-CITY-HISTORY-2.1','source':'Saudi Central Bank (SAMA) weekly POS via KAPSARC distribution',
        'scientific_boundary':'City Total rows only; no sector taxonomy is used.',
        'rows':int(len(panel)),'weeks':int(panel.week_start.nunique()),'cities':sorted(panel.city.unique().tolist()),'city_count':int(panel.city.nunique()),
        'date_start':str(panel.week_start.min().date()),'date_end':str(panel.week_start.max().date()),
        'max_gap_days_by_city':gaps,'coverage_per_week':{'min':int(weeks.min()),'median':float(weeks.median()),'max':int(weeks.max())},
        'checks':checks,'all_checks_passed':bool(all(checks.values())),
    }
    panel.to_csv(CITY,index=False); AUDIT.write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
    print(json.dumps(audit,indent=2,default=str))
    if not audit['all_checks_passed']: raise RuntimeError('SAMA city history quality gate failed')

if __name__=='__main__': main()
