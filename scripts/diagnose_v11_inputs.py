from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCES={
    'merchant_v7_1':ROOT/'data/merchant_v7_1/merchant_feature_panel_v7_1.csv',
    'saudi_sector_v1_5':ROOT/'data/saudi_v1_5/saudi_sector_daily_panel_v1_5.csv.gz',
}
OUT=ROOT/'reports/v11_input_diagnostics'
OUT.mkdir(parents=True,exist_ok=True)

def inspect(path):
    d=pd.read_csv(path,nrows=None)
    info={'rows':len(d),'columns':len(d.columns),'column_names':list(d.columns)}
    date_cols=[]
    for c in d.columns:
        if 'date' in c.lower() or 'week' in c.lower() or 'month' in c.lower():
            date_cols.append(c)
    info['date_like_columns']=date_cols
    time_ranges={}
    for c in date_cols[:10]:
        q=pd.to_datetime(d[c],errors='coerce')
        if q.notna().any(): time_ranges[c]={'min':str(q.min()),'max':str(q.max()),'unique':int(q.nunique())}
    info['time_ranges']=time_ranges
    keys=['sale','revenue','value','amount','invoice','transaction','customer','product','sector','category','sama','pos','risk','ratio','baseline','future','merchant','city','region']
    info['relevant_columns']={k:[c for c in d.columns if k in c.lower()] for k in keys}
    # Numeric variability to identify usable signals.
    numeric=d.select_dtypes(include=[np.number])
    vars=[]
    for c in numeric.columns:
        x=pd.to_numeric(numeric[c],errors='coerce')
        vars.append({'column':c,'non_null':int(x.notna().sum()),'unique':int(x.nunique()),'mean':float(x.mean()) if x.notna().any() else None,'std':float(x.std()) if x.notna().any() else None})
    vars.sort(key=lambda z:(z['std'] or 0),reverse=True)
    info['top_numeric_by_std']=vars[:40]
    return info

report={name:inspect(path) for name,path in SOURCES.items()}
(OUT/'diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
lines=['# V11 Input Diagnostics','']
for name,info in report.items():
    lines += [f'## {name}',f'- Rows: **{info["rows"]}**',f'- Columns: **{info["columns"]}**',f'- Date-like columns: **{info["date_like_columns"]}**',f'- Time ranges: **{info["time_ranges"]}**','', '### Relevant columns']
    for k,cols in info['relevant_columns'].items():
        if cols: lines.append(f'- {k}: {cols[:30]}')
    lines.append('')
(OUT/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print((OUT/'summary.md').read_text())
