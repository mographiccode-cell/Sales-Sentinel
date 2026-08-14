from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'/'sama_city_v2_1'; OUT.mkdir(parents=True,exist_ok=True)
URL='https://www.sama.gov.sa/en-US/Statistics/Indices/POS_EN/Weekly_Points_of_Sale_Transactions_Report_1st_Aug_2026.pdf'
NUM=re.compile(r'-?\d[\d,]*(?:\.\d+)?')


def ascii_prefix(line:str):
    # English label appears before Arabic text / first numeric cell.
    m=re.search(r'\d',line)
    prefix=line[:m.start()] if m else line
    prefix=prefix.encode('ascii','ignore').decode('ascii')
    prefix=re.sub(r'\s+',' ',prefix).strip(' -–—|')
    return prefix


def main():
    r=requests.get(URL,timeout=60,headers={'User-Agent':'Mozilla/5.0 Sales-Sentinel academic verifier'}); r.raise_for_status()
    p=Path('/tmp/city.pdf'); p.write_bytes(r.content)
    text=subprocess.run(['pdftotext','-layout',str(p),'-'],capture_output=True,text=True,check=True).stdout
    lines=text.splitlines(); marker=[i for i,l in enumerate(lines) if 'Table 2.1:' in l]
    if not marker: raise RuntimeError('Table 2.1 marker not found')
    start=marker[-1] if len(marker)>1 else marker[0]
    # Stop at next numbered table marker after the city table begins.
    end=len(lines)
    for j in range(start+1,len(lines)):
        if re.search(r'Table\s+(?:2\.2|3(?:\.\d+)?)\s*:',lines[j],re.I):
            end=j; break
    candidates=[]
    for i in range(start+1,end):
        vals=NUM.findall(lines[i])
        if len(vals)>=8:
            label=ascii_prefix(lines[i])
            if label and not label.lower().startswith(('number of','value of','weekly change')):
                candidates.append({'line':i+1,'label':label,'numeric_count':len(vals),'text':lines[i]})
    report={'url':URL,'marker_lines':[x+1 for x in marker],'selected_marker':start+1,'end_line':end+1,'candidate_rows':candidates}
    (OUT/'observed_city_labels.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'markers':report['marker_lines'],'selected_marker':start+1,'candidate_labels':[x['label'] for x in candidates]},indent=2))

if __name__=='__main__': main()
