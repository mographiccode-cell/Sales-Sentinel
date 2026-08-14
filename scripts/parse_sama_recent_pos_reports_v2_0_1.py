from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import parse_sama_recent_pos_reports_v2_0 as p

ROOT=Path(__file__).resolve().parents[1]
PDF_DIR=ROOT/'artifacts'/'sama_recent_v2_0'
REPORT_DIR=ROOT/'reports'/'sama_recent_v2_0'
OUT=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2025_2026_holdout.csv'
EXTENDED=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2020_2026_extended.csv'
OLD=ROOT/'data'/'sama_pos'/'sama_sector_weekly_value_count_2020_2025.csv'
AUDIT=REPORT_DIR/'parse_audit.json'
CUTOFF=pd.Timestamp('2025-07-07')


def main():
    paths=sorted(PDF_DIR.glob('*.pdf')); rows=[]; files=[]; failures=[]
    for path in paths:
        try:
            rr,info=p.parse_pdf(path); rows.extend(rr); files.append(info)
        except Exception as exc: failures.append({'file':path.name,'error':repr(exc)})
    if not rows: raise RuntimeError('No sector rows parsed')
    raw=pd.DataFrame(rows); raw['week_start']=pd.to_datetime(raw.week_start); raw['week_end']=pd.to_datetime(raw.week_end)

    # The sealed evaluation period begins strictly after the old development source (2025-07-06).
    # Older PDFs may be useful archive anchors but are NOT part of this holdout quality calculation.
    hraw=raw[raw.week_start>=CUTOFF].copy()
    spread=hraw.groupby(['week_start','sector']).agg(
        n=('value_thousand_sar','size'),
        value_min=('value_thousand_sar','min'),value_max=('value_thousand_sar','max'),
        count_min=('transaction_count_thousand','min'),count_max=('transaction_count_thousand','max'),
    ).reset_index()
    spread['value_rel_spread']=(spread.value_max-spread.value_min)/spread.value_max.replace(0,np.nan)
    spread['count_rel_spread']=(spread.count_max-spread.count_min)/spread.count_max.replace(0,np.nan)

    dedup=(hraw.sort_values(['week_start','sector','report_latest_week_end'])
             .drop_duplicates(['week_start','sector'],keep='last')
             .sort_values(['sector','week_start']).reset_index(drop=True))
    weeks=sorted(pd.to_datetime(dedup.week_start.unique()))
    week_gaps=pd.Series(weeks).diff().dt.days.dropna()
    coverage=dedup.groupby('week_start').sector.nunique()

    conflicts=spread[(spread.value_rel_spread>.01)|(spread.count_rel_spread>.01)].copy()
    conflict_records=[]
    for _,r in conflicts.sort_values(['week_start','sector']).head(20).iterrows():
        q=hraw[(hraw.week_start==r.week_start)&(hraw.sector==r.sector)].sort_values('report_latest_week_end')
        conflict_records.append({
            'week_start':str(r.week_start.date()),'sector':r.sector,
            'value_relative_spread':float(r.value_rel_spread),'count_relative_spread':float(r.count_rel_spread),
            'observations':q[['source_pdf','value_thousand_sar','transaction_count_thousand']].to_dict('records'),
        })

    checks={
        'at_least_20_pdfs_parsed':len(files)>=20,
        'pdf_failure_rate_below_10pct':len(failures)/max(len(paths),1)<.10,
        'at_least_12_safe_sectors':dedup.sector.nunique()>=12,
        'at_least_50_holdout_weeks':dedup.week_start.nunique()>=50,
        'starts_immediately_after_development':dedup.week_start.min()==pd.Timestamp('2025-07-13'),
        'covers_2026_august':dedup.week_start.max()>=pd.Timestamp('2026-08-01'),
        'strict_weekly_continuity':bool(len(week_gaps)>0 and (week_gaps==7).all()),
        'no_duplicate_week_sector':not dedup.duplicated(['week_start','sector']).any(),
        'positive_values':bool((dedup.value_thousand_sar>0).all()),
        'positive_counts':bool((dedup.transaction_count_thousand>0).all()),
        'every_week_has_all_safe_sectors':bool(coverage.nunique()==1 and coverage.iloc[0]==dedup.sector.nunique()),
        'holdout_overlap_value_revisions_below_1pct':float(spread.value_rel_spread.fillna(0).max())<.01,
        'holdout_overlap_count_revisions_below_1pct':float(spread.count_rel_spread.fillna(0).max())<.01,
        'ambiguous_miscellaneous_sector_excluded':'Miscellaneous Goods and Services' not in set(dedup.sector),
    }

    audit={
        'version':'SAMA-RECENT-HOLDOUT-2.0.1','source_boundary':'Official SAMA weekly POS PDFs; safe cross-taxonomy mappings only. Overlap consistency is evaluated only inside the sealed post-development holdout.',
        'pdfs_found':len(paths),'pdfs_parsed':len(files),'pdf_failures':failures,
        'holdout_rows':int(len(dedup)),'holdout_weeks':int(dedup.week_start.nunique()),'holdout_sector_count':int(dedup.sector.nunique()),
        'holdout_sectors':sorted(dedup.sector.unique().tolist()),
        'date_start':str(dedup.week_start.min().date()),'date_end':str(dedup.week_start.max().date()),
        'week_sector_coverage':{'min':int(coverage.min()),'median':float(coverage.median()),'max':int(coverage.max())},
        'max_holdout_overlap_value_relative_spread':float(spread.value_rel_spread.fillna(0).max()),
        'max_holdout_overlap_count_relative_spread':float(spread.count_rel_spread.fillna(0).max()),
        'holdout_overlap_conflicts_over_1pct':conflict_records,
        'checks':checks,'all_checks_passed':bool(all(checks.values())),
    }
    AUDIT.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    dedup[['week_start','week_end','sector','value_thousand_sar','transaction_count_thousand','source_pdf','source_activity_label']].to_csv(OUT,index=False)

    old=pd.read_csv(OLD,parse_dates=['week_start','week_end'])
    fresh=dedup[dedup.week_start>old.week_start.max()][['week_start','week_end','sector','value_thousand_sar','transaction_count_thousand']]
    ext=pd.concat([old,fresh],ignore_index=True).sort_values(['sector','week_start']).drop_duplicates(['week_start','sector'],keep='first')
    ext.to_csv(EXTENDED,index=False)

    print(json.dumps({
        'version':audit['version'],'holdout_rows':audit['holdout_rows'],'holdout_weeks':audit['holdout_weeks'],
        'holdout_sector_count':audit['holdout_sector_count'],'date_start':audit['date_start'],'date_end':audit['date_end'],
        'max_value_revision':audit['max_holdout_overlap_value_relative_spread'],'max_count_revision':audit['max_holdout_overlap_count_relative_spread'],
        'conflicts':conflict_records[:5],'checks':checks,'all_checks_passed':audit['all_checks_passed'],
    },indent=2))
    if not audit['all_checks_passed']: raise RuntimeError('SAMA sealed holdout v2.0.1 quality gate failed')

if __name__=='__main__': main()
