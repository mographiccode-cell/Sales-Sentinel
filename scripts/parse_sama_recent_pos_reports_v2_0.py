from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / 'artifacts' / 'sama_recent_v2_0'
REPORT_DIR = ROOT / 'reports' / 'sama_recent_v2_0'
OUT = ROOT / 'data' / 'sama_pos' / 'sama_sector_weekly_value_count_2025_2026_holdout.csv'
EXTENDED = ROOT / 'data' / 'sama_pos' / 'sama_sector_weekly_value_count_2020_2026_extended.csv'
OLD = ROOT / 'data' / 'sama_pos' / 'sama_sector_weekly_value_count_2020_2025.csv'
PARSE_REPORT = REPORT_DIR / 'parse_audit.json'

# Only semantic mappings that are defensible across the SAMA taxonomy revision are used.
ALIASES = {
    'Beverage and Food': ['Beverage and Food', 'Food & Beverages', 'Food and Beverages'],
    'Clothing and Footwear': ['Clothing and Footwear', 'Apparel, Clothing & Accessories', 'Apparel Clothing & Accessories'],
    'Construction & Building Materials': ['Construction & Building Materials'],
    'Education': ['Education'],
    'Electronic & Electric Devices': ['Electronic & Electric Devices'],
    'Furniture': ['Furniture', 'Furniture & Home Supplies'],
    'Gas Stations': ['Gas Stations'],
    'Health': ['Health'],
    'Hotels': ['Hotels'],
    'Jewelry': ['Jewelry'],
    # 'Miscellaneous Goods and Services' intentionally has no 2026 alias: the revised taxonomy splits it.
    'Other': ['Other', 'Others'],
    'Public Utilities': ['Public Utilities', 'Public Utilities & Services'],
    'Recreation and Culture': ['Recreation and Culture', 'Recreation & Culture'],
    'Restaurants & Café': ['Restaurants & Café', 'Restaurants & Cafe', 'Restaurants & Cafés', 'Restaurants & Cafes'],
    'Telecommunication': ['Telecommunication'],
    'Transportation': ['Transportation'],
}

DATE_RANGE_RE = re.compile(r'(\d{1,2}\s+[A-Za-z]{3},?\s*\d{2})\s*-\s*(\d{1,2}\s+[A-Za-z]{3},?\s*\d{2})')
NUM_RE = re.compile(r'-?\d[\d,]*(?:\.\d+)?')


def norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.replace('&', ' and ')
    s = re.sub(r'[^a-zA-Z0-9]+', ' ', s).strip().lower()
    return re.sub(r'\s+', ' ', s)

ALIAS_NORM = {norm(alias): canon for canon, aliases in ALIASES.items() for alias in aliases}


def parse_date(s: str) -> pd.Timestamp:
    s = re.sub(r'\s+', ' ', s.strip()).replace(' ,', ',')
    for fmt in ('%d %b,%y', '%d %b, %y', '%d %b %y'):
        try:
            return pd.Timestamp(datetime.strptime(s, fmt))
        except ValueError:
            pass
    raise ValueError(f'Unparsed SAMA date: {s!r}')


def pdf_text(path: Path) -> str:
    proc = subprocess.run(['pdftotext', '-layout', str(path), '-'], capture_output=True, text=True, check=True)
    return proc.stdout


def numeric_row(lines: list[str], idx: int):
    # In SAMA layouts the numeric row is normally immediately above the English activity name.
    for j in range(idx - 1, max(-1, idx - 5), -1):
        toks = NUM_RE.findall(lines[j])
        if len(toks) >= 8:
            vals = [float(x.replace(',', '')) for x in toks]
            return vals, lines[j]
    return None, None


def parse_pdf(path: Path):
    text = pdf_text(path)
    first_table = text.split('Table 2.1:', 1)[0]
    ranges = []
    for a, b in DATE_RANGE_RE.findall(first_table):
        pair = (parse_date(a), parse_date(b))
        if pair not in ranges:
            ranges.append(pair)
        if len(ranges) == 4:
            break
    if len(ranges) != 4:
        raise RuntimeError(f'{path.name}: expected 4 week ranges, got {len(ranges)}')

    lines = first_table.splitlines()
    rows = []
    found = {}
    for i, line in enumerate(lines):
        key = norm(line)
        canon = ALIAS_NORM.get(key)
        if not canon or canon in found:
            continue
        vals, raw = numeric_row(lines, i)
        if vals is None:
            continue
        # Four (count,value) pairs followed by weekly-change columns.
        for k, (ws, we) in enumerate(ranges):
            count = vals[2*k]
            value = vals[2*k+1]
            rows.append({
                'week_start': ws.normalize(), 'week_end': we.normalize(), 'sector': canon,
                'value_thousand_sar': value, 'transaction_count_thousand': count,
                'source_pdf': path.name, 'source_activity_label': line.strip(),
                'report_latest_week_end': ranges[-1][1].normalize(),
            })
        found[canon] = {'label': line.strip(), 'numeric_line': raw}
    return rows, {'file': path.name, 'week_ranges': [[str(a.date()), str(b.date())] for a,b in ranges], 'mapped_sectors': sorted(found), 'mapped_sector_count': len(found)}


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(PDF_DIR.glob('*.pdf'))
    if len(paths) < 20:
        raise RuntimeError(f'Need >=20 downloaded SAMA PDFs before parsing; found {len(paths)}')

    all_rows=[]; files=[]; failures=[]
    for path in paths:
        try:
            rows, info = parse_pdf(path)
            all_rows.extend(rows); files.append(info)
        except Exception as exc:
            failures.append({'file':path.name,'error':repr(exc)})

    if not all_rows:
        raise RuntimeError('No SAMA sector rows parsed')
    raw=pd.DataFrame(all_rows)
    raw['week_start']=pd.to_datetime(raw.week_start); raw['week_end']=pd.to_datetime(raw.week_end)

    # Overlapping 4-week reports should agree. Quantify any revisions, then keep the latest published report.
    spread = raw.groupby(['week_start','sector']).agg(
        n=('value_thousand_sar','size'),
        value_min=('value_thousand_sar','min'), value_max=('value_thousand_sar','max'),
        count_min=('transaction_count_thousand','min'), count_max=('transaction_count_thousand','max'),
    ).reset_index()
    spread['value_rel_spread']=(spread.value_max-spread.value_min)/spread.value_max.replace(0,np.nan)
    spread['count_rel_spread']=(spread.count_max-spread.count_min)/spread.count_max.replace(0,np.nan)

    dedup=(raw.sort_values(['week_start','sector','report_latest_week_end'])
             .drop_duplicates(['week_start','sector'],keep='last')
             .sort_values(['sector','week_start']).reset_index(drop=True))
    dedup=dedup[dedup.week_start>=pd.Timestamp('2025-07-07')].copy()

    weeks=dedup.groupby('week_start').sector.nunique()
    checks={
        'at_least_20_pdfs_parsed': len(files)>=20,
        'pdf_failure_rate_below_10pct': len(failures)/max(len(paths),1) < .10,
        'at_least_12_safe_sectors': dedup.sector.nunique()>=12,
        'at_least_40_holdout_weeks': dedup.week_start.nunique()>=40,
        'covers_2026': dedup.week_start.max()>=pd.Timestamp('2026-07-01'),
        'no_duplicate_week_sector': not dedup.duplicated(['week_start','sector']).any(),
        'positive_values': bool((dedup.value_thousand_sar>0).all()),
        'positive_counts': bool((dedup.transaction_count_thousand>0).all()),
        'median_week_sector_coverage_at_least_12': float(weeks.median())>=12,
        'overlap_value_revisions_below_1pct': float(spread.value_rel_spread.fillna(0).max())<.01,
        'overlap_count_revisions_below_1pct': float(spread.count_rel_spread.fillna(0).max())<.01,
        'ambiguous_miscellaneous_sector_excluded': 'Miscellaneous Goods and Services' not in set(dedup.sector),
    }

    audit={
        'version':'SAMA-RECENT-HOLDOUT-2.0',
        'source_boundary':'Official SAMA weekly POS PDFs. Only safe cross-taxonomy sector mappings are retained.',
        'pdfs_found':len(paths),'pdfs_parsed':len(files),'pdf_failures':failures,
        'parsed_files':files,
        'holdout_rows':len(dedup),'holdout_weeks':int(dedup.week_start.nunique()),
        'holdout_sectors':sorted(dedup.sector.unique().tolist()),
        'holdout_sector_count':int(dedup.sector.nunique()),
        'date_start':str(dedup.week_start.min().date()) if len(dedup) else None,
        'date_end':str(dedup.week_start.max().date()) if len(dedup) else None,
        'week_sector_coverage':{'min':int(weeks.min()) if len(weeks) else 0,'median':float(weeks.median()) if len(weeks) else 0,'max':int(weeks.max()) if len(weeks) else 0},
        'max_overlap_value_relative_spread':float(spread.value_rel_spread.fillna(0).max()),
        'max_overlap_count_relative_spread':float(spread.count_rel_spread.fillna(0).max()),
        'checks':checks,'all_checks_passed':bool(all(checks.values())),
    }
    PARSE_REPORT.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    dedup[['week_start','week_end','sector','value_thousand_sar','transaction_count_thousand','source_pdf','source_activity_label']].to_csv(OUT,index=False)

    old=pd.read_csv(OLD,parse_dates=['week_start','week_end'])
    # Fresh PDF data never overwrites the historical development set; append only strictly later weeks.
    fresh=dedup[dedup.week_start>old.week_start.max()][['week_start','week_end','sector','value_thousand_sar','transaction_count_thousand']]
    ext=pd.concat([old,fresh],ignore_index=True).sort_values(['sector','week_start']).drop_duplicates(['week_start','sector'],keep='first')
    ext.to_csv(EXTENDED,index=False)

    print(json.dumps({k:audit[k] for k in ['pdfs_found','pdfs_parsed','holdout_rows','holdout_weeks','holdout_sector_count','date_start','date_end','week_sector_coverage','checks','all_checks_passed']},indent=2))
    if not audit['all_checks_passed']:
        raise RuntimeError('SAMA recent holdout quality gate failed')


if __name__=='__main__':
    main()
