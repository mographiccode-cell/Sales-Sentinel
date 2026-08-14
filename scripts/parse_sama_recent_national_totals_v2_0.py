from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import parse_sama_recent_pos_reports_v2_0 as sector_parser

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / 'artifacts' / 'sama_recent_v2_0'
REPORT_DIR = ROOT / 'reports' / 'sama_recent_v2_0'
HOLDOUT = ROOT / 'data' / 'sama_pos' / 'sama_national_weekly_value_count_2025_2026_holdout.csv'
EXT_VALUE = ROOT / 'data' / 'sama_pos' / 'sama_pos_national_weekly_value_2020_2026_extended.csv'
EXT_COUNT = ROOT / 'data' / 'sama_pos' / 'sama_pos_national_weekly_count_2020_2026_extended.csv'
OLD_VALUE = ROOT / 'data' / 'sama_pos' / 'sama_pos_national_weekly_value_2020_2025.csv'
OLD_COUNT = ROOT / 'data' / 'sama_pos' / 'sama_pos_national_weekly_count_2020_2025.csv'
AUDIT = REPORT_DIR / 'national_parse_audit.json'


def parse_total(path: Path):
    text = sector_parser.pdf_text(path)
    first = text.split('Table 2.1:', 1)[0]
    ranges=[]
    for a,b in sector_parser.DATE_RANGE_RE.findall(first):
        pair=(sector_parser.parse_date(a), sector_parser.parse_date(b))
        if pair not in ranges: ranges.append(pair)
        if len(ranges)==4: break
    if len(ranges)!=4:
        raise RuntimeError(f'{path.name}: four weekly ranges not found')
    lines=first.splitlines()
    candidates=[]
    for i,line in enumerate(lines):
        if sector_parser.norm(line) != 'total':
            continue
        vals, raw = sector_parser.numeric_row(lines,i)
        if vals is not None and len(vals)>=8:
            candidates.append((i,vals,raw,line.strip()))
    if not candidates:
        raise RuntimeError(f'{path.name}: total row not found')
    # The national Total is the last top-level row before Table 2.1.
    _,vals,raw,label=candidates[-1]
    rows=[]
    for k,(ws,we) in enumerate(ranges):
        rows.append({
            'week_start':ws.normalize(),'week_end':we.normalize(),
            'value_thousand_sar':float(vals[2*k+1]),
            'transaction_count':float(vals[2*k]),
            'source_pdf':path.name,'source_label':label,
            'report_latest_week_end':ranges[-1][1].normalize(),
        })
    return rows


def main():
    paths=sorted(PDF_DIR.glob('*.pdf'))
    rows=[]; failures=[]
    for path in paths:
        try: rows.extend(parse_total(path))
        except Exception as exc: failures.append({'file':path.name,'error':repr(exc)})
    if not rows: raise RuntimeError('No national totals parsed')
    raw=pd.DataFrame(rows)
    raw['week_start']=pd.to_datetime(raw.week_start); raw['week_end']=pd.to_datetime(raw.week_end)
    spread=raw.groupby('week_start').agg(
        value_min=('value_thousand_sar','min'),value_max=('value_thousand_sar','max'),
        count_min=('transaction_count','min'),count_max=('transaction_count','max'),n=('value_thousand_sar','size')
    ).reset_index()
    spread['value_rel_spread']=(spread.value_max-spread.value_min)/spread.value_max.replace(0,np.nan)
    spread['count_rel_spread']=(spread.count_max-spread.count_min)/spread.count_max.replace(0,np.nan)
    d=(raw.sort_values(['week_start','report_latest_week_end'])
         .drop_duplicates('week_start',keep='last').sort_values('week_start').reset_index(drop=True))
    d=d[d.week_start>=pd.Timestamp('2025-07-07')].copy()
    checks={
        'at_least_40_weeks':d.week_start.nunique()>=40,
        'covers_2026_july':d.week_start.max()>=pd.Timestamp('2026-07-01'),
        'no_duplicate_weeks':not d.week_start.duplicated().any(),
        'positive_values':bool((d.value_thousand_sar>0).all()),
        'positive_counts':bool((d.transaction_count>0).all()),
        'overlap_value_revisions_below_1pct':float(spread.value_rel_spread.fillna(0).max())<.01,
        'overlap_count_revisions_below_1pct':float(spread.count_rel_spread.fillna(0).max())<.01,
        'pdf_parse_failure_below_10pct':len(failures)/max(len(paths),1)<.10,
    }
    d[['week_start','week_end','value_thousand_sar','transaction_count','source_pdf']].to_csv(HOLDOUT,index=False)

    ov=pd.read_csv(OLD_VALUE,parse_dates=['week_start','week_end'])
    oc=pd.read_csv(OLD_COUNT,parse_dates=['week_start','week_end'])
    fresh=d[d.week_start>ov.week_start.max()].copy()
    ev=pd.concat([
        ov,
        fresh.assign(source='Saudi Central Bank (SAMA) official weekly POS PDF',data_status='REAL_OFFICIAL_AGGREGATE')[['week_start','week_end','value_thousand_sar','source','data_status']]
    ],ignore_index=True).sort_values('week_start').drop_duplicates('week_start',keep='first')
    ec=pd.concat([
        oc,
        fresh.rename(columns={'transaction_count':'transaction_count'}).assign(source='Saudi Central Bank (SAMA) official weekly POS PDF',data_status='REAL_OFFICIAL_AGGREGATE')[['week_start','week_end','transaction_count','source','data_status']]
    ],ignore_index=True).sort_values('week_start').drop_duplicates('week_start',keep='first')
    ev.to_csv(EXT_VALUE,index=False); ec.to_csv(EXT_COUNT,index=False)

    audit={
        'version':'SAMA-NATIONAL-HOLDOUT-2.0','pdfs':len(paths),'failures':failures,
        'weeks':int(d.week_start.nunique()),'date_start':str(d.week_start.min().date()),'date_end':str(d.week_start.max().date()),
        'max_overlap_value_relative_spread':float(spread.value_rel_spread.fillna(0).max()),
        'max_overlap_count_relative_spread':float(spread.count_rel_spread.fillna(0).max()),
        'checks':checks,'all_checks_passed':bool(all(checks.values())),
    }
    AUDIT.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
    if not audit['all_checks_passed']:
        raise RuntimeError('National SAMA holdout quality gate failed')


if __name__=='__main__': main()
