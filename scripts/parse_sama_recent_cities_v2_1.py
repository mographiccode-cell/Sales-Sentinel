from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import parse_sama_recent_pos_reports_v2_0 as common

ROOT=Path(__file__).resolve().parents[1]
PDF_DIR=ROOT/'artifacts'/'sama_recent_v2_0'
HISTORY=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
OUT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2025_2026_holdout.csv'
EXT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
REPORT=ROOT/'reports'/'sama_city_v2_1'; REPORT.mkdir(parents=True,exist_ok=True)
AUDIT=REPORT/'fresh_holdout_audit.json'
CUTOFF=pd.Timestamp('2025-07-07')


def parse_city_pdf(path:Path,cities:list[str]):
    text=common.pdf_text(path)
    if 'Table 2.1:' not in text:
        raise RuntimeError(f'{path.name}: Table 2.1 not found')
    seg=text.split('Table 2.1:',1)[1]
    # Stop before any next table so similarly named text outside the city table cannot match.
    for marker in ('Table 3','Table 2.2','Table 3.1'):
        if marker in seg: seg=seg.split(marker,1)[0]
    ranges=[]
    for a,b in common.DATE_RANGE_RE.findall(seg):
        pair=(common.parse_date(a),common.parse_date(b))
        if pair not in ranges: ranges.append(pair)
        if len(ranges)==4: break
    if len(ranges)!=4:
        raise RuntimeError(f'{path.name}: expected 4 city-table week ranges, got {len(ranges)}')
    lookup={common.norm(c):c for c in cities}; found={}; rows=[]; lines=seg.splitlines()
    for i,line in enumerate(lines):
        key=common.norm(line)
        city=lookup.get(key)
        if not city or city in found: continue
        vals,raw=common.numeric_row(lines,i)
        if vals is None or len(vals)<8: continue
        for k,(ws,we) in enumerate(ranges):
            rows.append({'week_start':ws.normalize(),'week_end':we.normalize(),'city':city,
                         'value_thousand_sar':float(vals[2*k+1]),'transaction_count_thousand':float(vals[2*k]),
                         'source_pdf':path.name,'source_city_label':line.strip(),'report_latest_week_end':ranges[-1][1].normalize()})
        found[city]={'label':line.strip(),'numeric':raw}
    return rows,{'file':path.name,'week_ranges':[[str(a.date()),str(b.date())] for a,b in ranges],'cities_found':sorted(found),'city_count':len(found)}


def main():
    hist=pd.read_csv(HISTORY,parse_dates=['week_start','week_end']); cities=sorted(hist.city.unique().tolist())
    paths=sorted(PDF_DIR.glob('*.pdf')); rows=[]; files=[]; failures=[]
    for path in paths:
        try:
            rr,info=parse_city_pdf(path,cities); rows.extend(rr); files.append(info)
        except Exception as exc: failures.append({'file':path.name,'error':repr(exc)})
    if not rows: raise RuntimeError('No city rows parsed from fresh SAMA PDFs')
    raw=pd.DataFrame(rows); raw.week_start=pd.to_datetime(raw.week_start); raw.week_end=pd.to_datetime(raw.week_end)
    hraw=raw[raw.week_start>=CUTOFF].copy()
    spread=hraw.groupby(['week_start','city']).agg(n=('value_thousand_sar','size'),value_min=('value_thousand_sar','min'),value_max=('value_thousand_sar','max'),count_min=('transaction_count_thousand','min'),count_max=('transaction_count_thousand','max')).reset_index()
    spread['value_rel_spread']=(spread.value_max-spread.value_min)/spread.value_max.replace(0,np.nan)
    spread['count_rel_spread']=(spread.count_max-spread.count_min)/spread.count_max.replace(0,np.nan)
    # City totals are not affected by sector taxonomy. Latest overlapping official report is authoritative if identical/revised minutely.
    d=(hraw.sort_values(['week_start','city','report_latest_week_end']).drop_duplicates(['week_start','city'],keep='last').sort_values(['city','week_start']).reset_index(drop=True))
    weeks=sorted(pd.to_datetime(d.week_start.unique())); gaps=pd.Series(weeks).diff().dt.days.dropna(); coverage=d.groupby('week_start').city.nunique()
    city_gaps={c:int(q.week_start.sort_values().diff().dt.days.dropna().max()) for c,q in d.groupby('city')}
    checks={
        'all_pdfs_parsed':len(failures)==0,
        'all_historical_cities_present':set(d.city.unique())==set(cities),
        'at_least_50_holdout_weeks':d.week_start.nunique()>=50,
        'starts_post_development':d.week_start.min()==pd.Timestamp('2025-07-13'),
        'covers_aug_2026':d.week_start.max()>=pd.Timestamp('2026-08-01'),
        'strict_global_weekly_continuity':bool(len(gaps)>0 and (gaps==7).all()),
        'strict_city_weekly_continuity':max(city_gaps.values())<=7,
        'constant_city_coverage':coverage.nunique()==1 and int(coverage.iloc[0])==len(cities),
        'no_duplicate_city_week':not d.duplicated(['week_start','city']).any(),
        'positive_values':bool((d.value_thousand_sar>0).all()),'positive_counts':bool((d.transaction_count_thousand>0).all()),
        'overlap_value_consistency_below_1pct':float(spread.value_rel_spread.fillna(0).max())<.01,
        'overlap_count_consistency_below_1pct':float(spread.count_rel_spread.fillna(0).max())<.01,
    }
    audit={'version':'SAMA-CITY-FRESH-HOLDOUT-2.1','scientific_boundary':'Official SAMA city-total weekly POS only; independent of sector taxonomy revision.',
           'pdfs':len(paths),'parsed_pdfs':len(files),'failures':failures,'rows':int(len(d)),'weeks':int(d.week_start.nunique()),'cities':cities,'city_count':len(cities),
           'date_start':str(d.week_start.min().date()),'date_end':str(d.week_start.max().date()),'coverage':{'min':int(coverage.min()),'median':float(coverage.median()),'max':int(coverage.max())},
           'max_overlap_value_relative_spread':float(spread.value_rel_spread.fillna(0).max()),'max_overlap_count_relative_spread':float(spread.count_rel_spread.fillna(0).max()),
           'checks':checks,'all_checks_passed':bool(all(checks.values()))}
    d[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand','source_pdf','source_city_label']].to_csv(OUT,index=False)
    fresh=d[d.week_start>hist.week_start.max()][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']]
    ext=pd.concat([hist[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']],fresh],ignore_index=True).sort_values(['city','week_start']).drop_duplicates(['week_start','city'],keep='first')
    ext.to_csv(EXT,index=False); AUDIT.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
    if not audit['all_checks_passed']: raise RuntimeError('Fresh SAMA city holdout quality gate failed')

if __name__=='__main__': main()
