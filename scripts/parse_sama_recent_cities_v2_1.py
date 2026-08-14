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

CITY_ALIASES={
    'ABHA':['Abha'],
    'BURAIDAH':['Buraidah','Buraydah'],
    'DAMMAM':['Dammam'],
    'HAIL':['Hail','Hael','Hayel'],
    'JEDDAH':['Jeddah'],
    'KHOBAR':['Khobar','Al-Khobar','Al Khobar','Al-Khubar','Al Khubar'],
    'MADINA':['Al-Madinah','Al Madinah','Madinah','Madina'],
    'MAKKAH':['Makkah','Mecca'],
    'OTHER':['Other','Others'],
    'RIYADH':['Riyadh'],
    'TABOUK':['Tabouk','Tabuk'],
}


def match_city(line:str,cities:set[str]):
    normalized=common.norm(line)
    for canonical,aliases in CITY_ALIASES.items():
        if canonical not in cities: continue
        for alias in aliases:
            a=common.norm(alias)
            if normalized==a or normalized.startswith(a+' '):
                return canonical,alias
    return None,None


def parse_city_pdf(path:Path,cities:list[str]):
    text=common.pdf_text(path)
    marker='Table 2.1:'
    if marker not in text:
        raise RuntimeError(f'{path.name}: Table 2.1 not found')
    # Table 2.1 spans multiple PDF pages. Extract its four date ranges near the marker,
    # then scan ALL later lines for explicit historical city aliases. The first matching
    # row for each city is its Table 2.1 city-total row; later tables cannot replace it.
    before,after=text.split(marker,1)
    header_window='\n'.join(after.splitlines()[:45])
    ranges=[]
    for a,b in common.DATE_RANGE_RE.findall(header_window):
        pair=(common.parse_date(a),common.parse_date(b))
        if pair not in ranges: ranges.append(pair)
        if len(ranges)==4: break
    if len(ranges)!=4:
        # Defensive fallback: the dates may be split farther down by PDF text extraction.
        for a,b in common.DATE_RANGE_RE.findall(after):
            pair=(common.parse_date(a),common.parse_date(b))
            if pair not in ranges: ranges.append(pair)
            if len(ranges)==4: break
    if len(ranges)!=4:
        raise RuntimeError(f'{path.name}: expected 4 city-table week ranges, got {len(ranges)}')

    city_set=set(cities); found={}; rows=[]
    for line in after.splitlines():
        city,alias=match_city(line,city_set)
        if not city or city in found: continue
        vals=[float(x.replace(',','')) for x in common.NUM_RE.findall(line)]
        if len(vals)<8: continue
        # Table 2.1 row: four (number transactions, value transactions) pairs,
        # followed by two weekly-change percentages.
        for k,(ws,we) in enumerate(ranges):
            rows.append({
                'week_start':ws.normalize(),'week_end':we.normalize(),'city':city,
                'value_thousand_sar':float(vals[2*k+1]),
                'transaction_count_thousand':float(vals[2*k]),
                'source_pdf':path.name,'source_city_label':line.strip(),'source_city_alias':alias,
                'report_latest_week_end':ranges[-1][1].normalize(),
            })
        found[city]={'alias':alias,'line':line.strip()}
        if len(found)==len(city_set): break
    missing=sorted(city_set-set(found))
    if missing:
        raise RuntimeError(f'{path.name}: missing historical city labels after complete multi-page scan: {missing}')
    return rows,{
        'file':path.name,
        'week_ranges':[[str(a.date()),str(b.date())] for a,b in ranges],
        'cities_found':sorted(found),'city_count':len(found),
        'matched_aliases':{c:v['alias'] for c,v in found.items()},
    }


def main():
    hist=pd.read_csv(HISTORY,parse_dates=['week_start','week_end']); cities=sorted(hist.city.unique().tolist())
    missing_alias_defs=sorted(set(cities)-set(CITY_ALIASES))
    if missing_alias_defs: raise RuntimeError(f'No explicit fresh-PDF alias definition for historical cities: {missing_alias_defs}')
    paths=sorted(PDF_DIR.glob('*.pdf')); rows=[]; files=[]; failures=[]
    for path in paths:
        try:
            rr,info=parse_city_pdf(path,cities); rows.extend(rr); files.append(info)
        except Exception as exc:
            failures.append({'file':path.name,'error':repr(exc)})
    if not rows:
        raise RuntimeError(f'No city rows parsed from fresh SAMA PDFs; failures={failures[:5]}')

    raw=pd.DataFrame(rows); raw.week_start=pd.to_datetime(raw.week_start); raw.week_end=pd.to_datetime(raw.week_end)
    hraw=raw[raw.week_start>=CUTOFF].copy()
    spread=hraw.groupby(['week_start','city']).agg(n=('value_thousand_sar','size'),value_min=('value_thousand_sar','min'),value_max=('value_thousand_sar','max'),count_min=('transaction_count_thousand','min'),count_max=('transaction_count_thousand','max')).reset_index()
    spread['value_rel_spread']=(spread.value_max-spread.value_min)/spread.value_max.replace(0,np.nan)
    spread['count_rel_spread']=(spread.count_max-spread.count_min)/spread.count_max.replace(0,np.nan)
    d=(hraw.sort_values(['week_start','city','report_latest_week_end']).drop_duplicates(['week_start','city'],keep='last').sort_values(['city','week_start']).reset_index(drop=True))
    weeks=sorted(pd.to_datetime(d.week_start.unique())); gaps=pd.Series(weeks).diff().dt.days.dropna(); coverage=d.groupby('week_start').city.nunique()
    city_week_sets={c:tuple(q.week_start.sort_values().tolist()) for c,q in d.groupby('city')}; ref=next(iter(city_week_sets.values())) if city_week_sets else tuple(); all_city_week_keys_identical=all(v==ref for v in city_week_sets.values())

    national_path=ROOT/'data'/'sama_pos'/'sama_national_weekly_value_count_2025_2026_holdout.csv'; reconciliation=None; reconciliation_checks=True
    if national_path.exists():
        nat=pd.read_csv(national_path,parse_dates=['week_start']); city_sum=d.groupby('week_start',as_index=False).agg(city_value=('value_thousand_sar','sum'),city_count=('transaction_count_thousand','sum'))
        rr=city_sum.merge(nat[['week_start','value_thousand_sar','transaction_count']],on='week_start',how='inner')
        if len(rr):
            rr['value_rel_diff']=(rr.city_value-rr.value_thousand_sar).abs()/rr.value_thousand_sar; rr['count_rel_diff']=(rr.city_count-rr.transaction_count).abs()/rr.transaction_count
            reconciliation={'weeks_compared':int(len(rr)),'max_value_relative_difference':float(rr.value_rel_diff.max()),'max_count_relative_difference':float(rr.count_rel_diff.max())}
            reconciliation_checks=(reconciliation['max_value_relative_difference']<.0002 and reconciliation['max_count_relative_difference']<.0002)

    checks={
        'pdf_parse_success_at_least_95pct':len(failures)/max(len(paths),1)<=.05,
        'all_historical_cities_present':set(d.city.unique())==set(cities),
        'at_least_50_holdout_weeks':d.week_start.nunique()>=50,
        'starts_post_development':d.week_start.min()==pd.Timestamp('2025-07-13'),
        'covers_aug_2026':d.week_start.max()>=pd.Timestamp('2026-08-01'),
        'strict_global_weekly_continuity':bool(len(gaps)>0 and (gaps==7).all()),
        'all_cities_share_identical_fresh_week_keys':all_city_week_keys_identical,
        'constant_city_coverage':coverage.nunique()==1 and int(coverage.iloc[0])==len(cities),
        'no_duplicate_city_week':not d.duplicated(['week_start','city']).any(),
        'positive_values':bool((d.value_thousand_sar>0).all()),'positive_counts':bool((d.transaction_count_thousand>0).all()),
        'overlap_value_consistency_below_1pct':float(spread.value_rel_spread.fillna(0).max())<.01,
        'overlap_count_consistency_below_1pct':float(spread.count_rel_spread.fillna(0).max())<.01,
        'fresh_city_sum_reconciles_to_national_when_available':bool(reconciliation_checks),
    }
    audit={'version':'SAMA-CITY-FRESH-HOLDOUT-2.1.3','scientific_boundary':'Official SAMA City Total weekly POS only; independent of sector taxonomy revision. Table 2.1 is scanned across all PDF pages with explicit historical-city aliases.',
           'pdfs':len(paths),'parsed_pdfs':len(files),'failures':failures,'rows':int(len(d)),'weeks':int(d.week_start.nunique()),'cities':cities,'city_count':len(cities),
           'date_start':str(d.week_start.min().date()),'date_end':str(d.week_start.max().date()),'coverage':{'min':int(coverage.min()),'median':float(coverage.median()),'max':int(coverage.max())},
           'max_overlap_value_relative_spread':float(spread.value_rel_spread.fillna(0).max()),'max_overlap_count_relative_spread':float(spread.count_rel_spread.fillna(0).max()),'national_reconciliation':reconciliation,
           'checks':checks,'all_checks_passed':bool(all(checks.values()))}
    d[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand','source_pdf','source_city_label','source_city_alias']].to_csv(OUT,index=False)
    fresh=d[d.week_start>hist.week_start.max()][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']]
    ext=pd.concat([hist[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']],fresh],ignore_index=True).sort_values(['city','week_start']).drop_duplicates(['week_start','city'],keep='first')
    ext.to_csv(EXT,index=False); AUDIT.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
    if not audit['all_checks_passed']: raise RuntimeError('Fresh SAMA city holdout quality gate failed')

if __name__=='__main__': main()
