from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import unquote

import requests

import scrape_sama_recent_pos_reports_v2_0 as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts'/'sama_recent_v2_0'
REPORT=ROOT/'reports'/'sama_recent_v2_0'/'gap_2025_manifest.json'
PAGE='https://www.sama.gov.sa/en-US/Statistics/Indices/pages/pos.aspx'


def main():
    s=requests.Session(); s.headers.update({'User-Agent':'Mozilla/5.0 Sales-Sentinel academic data verifier'})
    refs=set(); pages=[]
    # SAMA IDs around the known 4-Aug-2025 report. Probe each weekly ID with the official Statistics path.
    # Date is used only as a SharePoint paging anchor; invalid combinations are safely ignored.
    start=date(2025,7,7)
    for pid in range(276,291):
        for offset in range(-7,8,7):
            publish=start+timedelta(days=(pid-276)*7+offset)
            url=(f'{PAGE}?Paged=TRUE&p_ID={pid}&p_SAMAFilePublishDate={publish:%Y%m%d}+21%3A00%3A00'
                 '&p_SortBehavior=0&View=%7BECDECFC9-707B-4F26-9830-F3EA40503071%7D')
            try:
                r=s.get(url,timeout=30)
                if r.status_code!=200:
                    pages.append({'pid':pid,'publish':str(publish),'status':r.status_code,'refs':0}); continue
                found=base.extract_refs(r.text); refs |= found
                pages.append({'pid':pid,'publish':str(publish),'status':200,'refs':len(found),'final_url':r.url})
            except Exception as exc:
                pages.append({'pid':pid,'publish':str(publish),'error':repr(exc),'refs':0})

    candidates=sorted({base.normalize(x) for x in refs if '.pdf' in x.lower()})
    candidates=[u for u in candidates if '/statistics/indices/' in u.lower()]
    downloads=[]; before={p.name for p in OUT.glob('*.pdf')}
    for url in candidates:
        name=unquote(url.split('?')[0].split('/')[-1])
        if '2025' not in name: continue
        try:
            r=s.get(url,timeout=60)
            ok=r.status_code==200 and r.content.startswith(b'%PDF')
            row={'url':url,'name':name,'status':r.status_code,'bytes':len(r.content),'is_pdf':ok}
            if ok:
                path=OUT/name; path.write_bytes(r.content); row['sha256']=base.sha256(path)
            downloads.append(row)
        except Exception as exc: downloads.append({'url':url,'name':name,'error':repr(exc),'is_pdf':False})
    after={p.name for p in OUT.glob('*.pdf')}
    new=sorted(after-before)
    manifest={'pages_probed':len(pages),'page_probes':pages,'candidate_urls':len(candidates),'new_pdfs':new,'new_pdf_count':len(new),'total_pdf_count':len(after),'downloads':downloads}
    REPORT.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'new_pdf_count':len(new),'total_pdf_count':len(after),'new_pdfs':new},indent=2))
    # Do not fail when the main archive anchors already supplied the same weeks.

if __name__=='__main__': main()
