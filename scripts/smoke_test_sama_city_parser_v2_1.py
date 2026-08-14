from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

import parse_sama_recent_cities_v2_1 as parser

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'sama_city_v2_1'; OUT.mkdir(parents=True,exist_ok=True)
HISTORY=ROOT/'data'/'sama_pos'/'sama_city_weekly_value_count_2020_2025.csv'
URLS=[
 'https://www.sama.gov.sa/en-US/Statistics/Indices/POS_EN/POS_Report_26_Jul_2025.pdf',
 'https://www.sama.gov.sa/en-US/Statistics/Indices/POS_EN/Weekly_Points_of_Sale_Transactions_Report_1st_Aug_2026.pdf',
]

def main():
    cities=sorted(pd.read_csv(HISTORY).city.unique().tolist()); s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0 Sales-Sentinel academic verifier'
    results=[]
    for url in URLS:
        r=s.get(url,timeout=60); r.raise_for_status(); p=Path('/tmp')/url.split('/')[-1]; p.write_bytes(r.content)
        rows,info=parser.parse_city_pdf(p,cities)
        df=pd.DataFrame(rows)
        results.append({'file':p.name,'rows':len(df),'cities':sorted(df.city.unique().tolist()),'city_count':df.city.nunique(),'weeks':df.week_start.nunique(),'info':info})
    checks={'two_reports':len(results)==2,'44_rows_each':all(x['rows']==44 for x in results),'11_cities_each':all(x['city_count']==11 for x in results),'4_weeks_each':all(x['weeks']==4 for x in results)}
    report={'results':results,'checks':checks,'all_checks_passed':all(checks.values())}
    (OUT/'city_parser_smoke_test.json').write_text(json.dumps(report,indent=2,default=str),encoding='utf-8')
    print(json.dumps(report,indent=2,default=str))
    if not report['all_checks_passed']: raise RuntimeError('City parser smoke test failed')

if __name__=='__main__': main()
