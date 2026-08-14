from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import parse_sama_recent_pos_reports_v2_0 as p

ROOT=Path(__file__).resolve().parents[1]
PDF_DIR=ROOT/'artifacts'/'sama_recent_v2_0'
OUT=ROOT/'reports'/'sama_recent_v2_0'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'overlap_diagnostics.json'


def main():
    rows=[]; failures=[]
    for path in sorted(PDF_DIR.glob('*.pdf')):
        try:
            rr,_=p.parse_pdf(path); rows.extend(rr)
        except Exception as exc: failures.append({'file':path.name,'error':repr(exc)})
    raw=pd.DataFrame(rows); raw['week_start']=pd.to_datetime(raw.week_start)
    grp=raw.groupby(['week_start','sector']).agg(
        n=('value_thousand_sar','size'),
        value_min=('value_thousand_sar','min'),value_max=('value_thousand_sar','max'),
        count_min=('transaction_count_thousand','min'),count_max=('transaction_count_thousand','max'),
    ).reset_index()
    grp['value_rel_spread']=(grp.value_max-grp.value_min)/grp.value_max.replace(0,np.nan)
    grp['count_rel_spread']=(grp.count_max-grp.count_min)/grp.count_max.replace(0,np.nan)
    grp['worst']=grp[['value_rel_spread','count_rel_spread']].max(axis=1)
    top=grp.sort_values('worst',ascending=False).head(30)
    details=[]
    for _,r in top.iterrows():
        q=raw[(raw.week_start==r.week_start)&(raw.sector==r.sector)].sort_values('report_latest_week_end')
        details.append({
            'week_start':str(r.week_start.date()),'sector':r.sector,'n':int(r.n),
            'value_rel_spread':float(r.value_rel_spread),'count_rel_spread':float(r.count_rel_spread),
            'observations':q[['source_pdf','source_activity_label','value_thousand_sar','transaction_count_thousand','report_latest_week_end']].assign(report_latest_week_end=lambda z:z.report_latest_week_end.astype(str)).to_dict('records')
        })
    report={
        'pdfs':len(list(PDF_DIR.glob('*.pdf'))),'failures':failures,'parsed_rows':len(raw),
        'overlap_cells':int((grp.n>1).sum()),
        'value_spread_quantiles':{str(q):float(grp.value_rel_spread.fillna(0).quantile(q)) for q in [.5,.9,.95,.99,1.0]},
        'count_spread_quantiles':{str(q):float(grp.count_rel_spread.fillna(0).quantile(q)) for q in [.5,.9,.95,.99,1.0]},
        'cells_value_over_1pct':int((grp.value_rel_spread>.01).sum()),
        'cells_count_over_1pct':int((grp.count_rel_spread>.01).sum()),
        'top_conflicts':details,
    }
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
