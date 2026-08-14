from __future__ import annotations

import json
import subprocess
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'sama_city_v2_1'; OUT.mkdir(parents=True,exist_ok=True)
URLS=[
 'https://www.sama.gov.sa/en-US/Statistics/Indices/POS_EN/POS_Report_26_Jul_2025.pdf',
 'https://www.sama.gov.sa/en-US/Statistics/Indices/POS_EN/Weekly_Points_of_Sale_Transactions_Report_1st_Aug_2026.pdf',
]
TOKENS=['riyadh','jeddah','makkah','mecca','madina','madinah','dammam','khobar','abha','hail','buraidah','buraydah','tabuk','tabouk','other','table 2.1']

def main():
    s=requests.Session(); s.headers['User-Agent']='Mozilla/5.0 Sales-Sentinel academic verifier'
    docs=[]
    for url in URLS:
        r=s.get(url,timeout=60); row={'url':url,'status':r.status_code,'bytes':len(r.content),'is_pdf':r.content.startswith(b'%PDF')}
        if r.status_code!=200 or not row['is_pdf']:
            docs.append(row); continue
        p=Path('/tmp')/url.split('/')[-1]; p.write_bytes(r.content)
        txt=subprocess.run(['pdftotext','-layout',str(p),'-'],capture_output=True,text=True,check=True).stdout
        lines=txt.splitlines(); hits=[]
        for i,line in enumerate(lines):
            low=line.lower()
            if any(t in low for t in TOKENS):
                hits.append({'line':i+1,'text':line,'context':[{'line':j+1,'text':lines[j]} for j in range(max(0,i-4),min(len(lines),i+5))]})
        row.update({'line_count':len(lines),'hits':hits[:120]}); docs.append(row)
    report={'documents':docs}; (OUT/'quick_pdf_layout.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
