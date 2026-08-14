from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import parse_sama_recent_pos_reports_v2_0 as common
import parse_sama_recent_national_totals_v2_0 as national_parser

ROOT=Path(__file__).resolve().parents[1]
PDF_DIR=ROOT/'artifacts'/'sama_recent_v2_0'
HISTORY=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
OUT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2025_2026_holdout.csv'
EXT=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2026_extended.csv'
REPORT=ROOT/'reports'/'sama_city_v2_1'; REPORT.mkdir(parents=True,exist_ok=True)
AUDIT=REPORT/'fresh_holdout_audit.json'
CUTOFF=pd.Timestamp('2025-07-07')

# The 10 named cities are semantically stable across the old machine-readable table and new PDF taxonomy.
# Historical OTHER means the residual of all other cities. New PDFs enumerate many more cities, so the new
# row labelled "Others" is NOT used as historical OTHER; we reconstruct the historical definition exactly
# as official national total minus these 10 named historical cities.
CITY_ALIASES={
    'ABHA':['Abha'],
    'BURAIDAH':['Buraidah','Buraydah'],
    'DAMMAM':['Dammam'],
    'HAIL':['Hail','Hael','Hayel'],
    'JEDDAH':['Jeddah'],
    'KHOBAR':['Khobar','Al-Khobar','Al Khobar','Al-Khubar','Al Khubar'],
    'MADINA':['Al-Madinah','Al Madinah','Madinah','Madina'],
    'MAKKAH':['Makkah','Mecca'],
    'RIYADH':['Riyadh'],
    'TABOUK':['Tabouk','Tabuk'],
}
NAMED_CITIES=sorted(CITY_ALIASES)


def match_city(line:str):
    normalized=common.norm(line)
    for canonical,aliases in CITY_ALIASES.items():
        for alias in aliases:
            a=common.norm(alias)
            if normalized==a or normalized.startswith(a+' '):
                return canonical,alias
    return None,None


def parse_named_city_pdf(path:Path):
    text=common.pdf_text(path); marker='Table 2.1:'
    if marker not in text: raise RuntimeError(f'{path.name}: Table 2.1 not found')
    _,after=text.split(marker,1)
    ranges=[]
    for a,b in common.DATE_RANGE_RE.findall('\n'.join(after.splitlines()[:60])):
        pair=(common.parse_date(a),common.parse_date(b))
        if pair not in ranges:ranges.append(pair)
        if len(ranges)==4:break
    if len(ranges)!=4:
        for a,b in common.DATE_RANGE_RE.findall(after):
            pair=(common.parse_date(a),common.parse_date(b))
            if pair not in ranges:ranges.append(pair)
            if len(ranges)==4:break
    if len(ranges)!=4: raise RuntimeError(f'{path.name}: expected four Table 2.1 week ranges; got {len(ranges)}')

    found={}; rows=[]
    for line in after.splitlines():
        city,alias=match_city(line)
        if not city or city in found:continue
        vals=[float(x.replace(',','')) for x in common.NUM_RE.findall(line)]
        if len(vals)<8:continue
        for k,(ws,we) in enumerate(ranges):
            rows.append({'week_start':ws.normalize(),'week_end':we.normalize(),'city':city,
                         'value_thousand_sar':float(vals[2*k+1]),'transaction_count_thousand':float(vals[2*k]),
                         'source_pdf':path.name,'source_city_label':line.strip(),'source_city_alias':alias,
                         'report_latest_week_end':ranges[-1][1].normalize()})
        found[city]=alias
        if len(found)==len(NAMED_CITIES):break
    missing=sorted(set(NAMED_CITIES)-set(found))
    if missing: raise RuntimeError(f'{path.name}: missing named historical cities: {missing}')
    return rows,{'file':path.name,'city_count':len(found),'matched_aliases':found,
                 'week_ranges':[[str(a.date()),str(b.date())] for a,b in ranges]}


def main():
    hist=pd.read_csv(HISTORY,parse_dates=['week_start','week_end'])
    historical_cities=sorted(hist.city.unique().tolist())
    if set(historical_cities)!=set(NAMED_CITIES+['OTHER']):
        raise RuntimeError(f'Historical city schema changed: {historical_cities}')

    paths=sorted(PDF_DIR.glob('*.pdf')); named_rows=[]; national_rows=[]; files=[]; failures=[]
    for path in paths:
        try:
            rr,info=parse_named_city_pdf(path)
            nr=national_parser.parse_total(path)
            named_rows.extend(rr); national_rows.extend(nr); files.append(info)
        except Exception as exc:
            failures.append({'file':path.name,'error':repr(exc)})
    if not named_rows or not national_rows:
        raise RuntimeError(f'No usable city/national rows parsed; failures={failures[:5]}')

    raw=pd.DataFrame(named_rows); raw.week_start=pd.to_datetime(raw.week_start); raw.week_end=pd.to_datetime(raw.week_end)
    natraw=pd.DataFrame(national_rows); natraw.week_start=pd.to_datetime(natraw.week_start); natraw.week_end=pd.to_datetime(natraw.week_end)
    raw=raw[raw.week_start>=CUTOFF].copy(); natraw=natraw[natraw.week_start>=CUTOFF].copy()

    # Verify overlapping official reports before taking the latest copy.
    city_spread=raw.groupby(['week_start','city']).agg(value_min=('value_thousand_sar','min'),value_max=('value_thousand_sar','max'),count_min=('transaction_count_thousand','min'),count_max=('transaction_count_thousand','max')).reset_index()
    city_spread['value_rel_spread']=(city_spread.value_max-city_spread.value_min)/city_spread.value_max.replace(0,np.nan)
    city_spread['count_rel_spread']=(city_spread.count_max-city_spread.count_min)/city_spread.count_max.replace(0,np.nan)
    nat_spread=natraw.groupby('week_start').agg(value_min=('value_thousand_sar','min'),value_max=('value_thousand_sar','max'),count_min=('transaction_count','min'),count_max=('transaction_count','max')).reset_index()
    nat_spread['value_rel_spread']=(nat_spread.value_max-nat_spread.value_min)/nat_spread.value_max.replace(0,np.nan)
    nat_spread['count_rel_spread']=(nat_spread.count_max-nat_spread.count_min)/nat_spread.count_max.replace(0,np.nan)

    named=(raw.sort_values(['week_start','city','report_latest_week_end']).drop_duplicates(['week_start','city'],keep='last'))
    nat=(natraw.sort_values(['week_start','report_latest_week_end']).drop_duplicates('week_start',keep='last'))
    sums=named.groupby(['week_start','week_end'],as_index=False).agg(named_value=('value_thousand_sar','sum'),named_count=('transaction_count_thousand','sum'))
    residual=nat[['week_start','week_end','value_thousand_sar','transaction_count','source_pdf']].merge(sums,on=['week_start','week_end'],how='inner',validate='one_to_one')
    residual['city']='OTHER'
    residual['value_thousand_sar']=residual.value_thousand_sar-residual.named_value
    residual['transaction_count_thousand']=residual.transaction_count-residual.named_count
    residual['source_city_label']='RECONSTRUCTED_HISTORICAL_OTHER = OFFICIAL_NATIONAL_TOTAL - 10_HISTORICAL_NAMED_CITIES'
    residual['source_city_alias']='RECONSTRUCTED_RESIDUAL'
    other=residual[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand','source_pdf','source_city_label','source_city_alias']]
    named=named[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand','source_pdf','source_city_label','source_city_alias']]
    d=pd.concat([named,other],ignore_index=True).sort_values(['city','week_start']).reset_index(drop=True)

    # Exact reconciliation is now a definition-preserving independent invariant.
    city_sum=d.groupby('week_start',as_index=False).agg(city_value=('value_thousand_sar','sum'),city_count=('transaction_count_thousand','sum'))
    rec=city_sum.merge(nat[['week_start','value_thousand_sar','transaction_count']],on='week_start',validate='one_to_one')
    rec['value_rel_diff']=(rec.city_value-rec.value_thousand_sar).abs()/rec.value_thousand_sar
    rec['count_rel_diff']=(rec.city_count-rec.transaction_count).abs()/rec.transaction_count

    weeks=sorted(pd.to_datetime(d.week_start.unique())); gaps=pd.Series(weeks).diff().dt.days.dropna(); coverage=d.groupby('week_start').city.nunique()
    actual_span=(weeks[-1]-weeks[0]).days if len(weeks)>1 else 0; expected_span=(len(weeks)-1)*7
    city_sets={c:tuple(q.week_start.sort_values()) for c,q in d.groupby('city')}; ref=next(iter(city_sets.values())); same_keys=all(v==ref for v in city_sets.values())
    holdout_files=[x for x in files if pd.Timestamp(x['week_ranges'][-1][1])>=CUTOFF]
    checks={
        'all_11_historical_city_definitions_present':set(d.city.unique())==set(historical_cities),
        'at_least_56_holdout_weeks':d.week_start.nunique()>=56,
        'starts_post_development':d.week_start.min()==pd.Timestamp('2025-07-13'),
        'covers_aug_2026':d.week_start.max()>=pd.Timestamp('2026-08-01'),
        'strict_weekly_sequence':bool(len(gaps)>0 and (gaps==7).all() and actual_span==expected_span),
        'all_cities_share_identical_week_keys':same_keys,
        'constant_11_city_coverage':coverage.nunique()==1 and int(coverage.iloc[0])==11,
        'no_duplicate_city_week':not d.duplicated(['week_start','city']).any(),
        'positive_values':bool((d.value_thousand_sar>0).all()),'positive_counts':bool((d.transaction_count_thousand>0).all()),
        'named_city_overlap_consistency_below_1pct':float(city_spread[['value_rel_spread','count_rel_spread']].fillna(0).to_numpy().max())<.01,
        'national_overlap_consistency_below_1pct':float(nat_spread[['value_rel_spread','count_rel_spread']].fillna(0).to_numpy().max())<.01,
        'fresh_city_sum_reconciles_to_national_below_0_01pct':float(max(rec.value_rel_diff.max(),rec.count_rel_diff.max()))<.0001,
        'other_residual_positive':bool((other.value_thousand_sar>0).all() and (other.transaction_count_thousand>0).all()),
    }
    audit={'version':'SAMA-CITY-FRESH-HOLDOUT-2.1.4','scientific_boundary':'Official SAMA City Total weekly POS. The historical OTHER definition is preserved after the 2025 city-table expansion by reconstructing OTHER as official national total minus the same 10 named historical cities.',
           'pdfs_downloaded':len(paths),'usable_report_files':len(files),'holdout_report_files':len(holdout_files),'failures':failures,
           'rows':int(len(d)),'weeks':int(d.week_start.nunique()),'city_count':int(d.city.nunique()),'cities':historical_cities,
           'date_start':str(d.week_start.min().date()),'date_end':str(d.week_start.max().date()),'week_gap_days':sorted(set(gaps.astype(int).tolist())),
           'coverage':{'min':int(coverage.min()),'median':float(coverage.median()),'max':int(coverage.max())},
           'national_reconciliation':{'max_value_relative_difference':float(rec.value_rel_diff.max()),'max_count_relative_difference':float(rec.count_rel_diff.max())},
           'max_named_overlap_value_relative_spread':float(city_spread.value_rel_spread.fillna(0).max()),'max_named_overlap_count_relative_spread':float(city_spread.count_rel_spread.fillna(0).max()),
           'max_national_overlap_value_relative_spread':float(nat_spread.value_rel_spread.fillna(0).max()),'max_national_overlap_count_relative_spread':float(nat_spread.count_rel_spread.fillna(0).max()),
           'checks':checks,'all_checks_passed':bool(all(checks.values()))}

    d.to_csv(OUT,index=False)
    fresh=d[d.week_start>hist.week_start.max()][['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']]
    ext=pd.concat([hist[['week_start','week_end','city','value_thousand_sar','transaction_count_thousand']],fresh],ignore_index=True).sort_values(['city','week_start']).drop_duplicates(['week_start','city'],keep='first')
    ext.to_csv(EXT,index=False); AUDIT.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
    if not audit['all_checks_passed']: raise RuntimeError('Fresh SAMA city holdout quality gate failed')

if __name__=='__main__':main()
