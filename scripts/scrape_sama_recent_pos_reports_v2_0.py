from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urljoin, unquote

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "sama_recent_v2_0"
REPORT = ROOT / "reports" / "sama_recent_v2_0"
OUT.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

BASE = "https://www.sama.gov.sa"
CUTOFF = "2025-07-07"

ANCHORS = [
    "https://www.sama.gov.sa/en-US/Statistics/Indices/pages/pos.aspx",
    "https://www.sama.gov.sa/en-US/Statistics/Indices/pages/pos.aspx?PageFirstRow=51&Paged=TRUE&PagedPrev=TRUE&View=%7BECDECFC9-707B-4F26-9830-F3EA40503071%7D&p_ID=317&p_SAMAFilePublishDate=20260427+21%3A00%3A00&p_SortBehavior=0",
    "https://www.sama.gov.sa/en-US/Statistics/Indices/pages/pos.aspx?PageFirstRow=331&Paged=TRUE&View=%7BECDECFC9-707B-4F26-9830-F3EA40503071%7D&p_ID=313&p_SAMAFilePublishDate=20260330+21%3A00%3A00&p_SortBehavior=0",
    "https://www.sama.gov.sa/en-US/Statistics/Indices/pages/pos.aspx?PageFirstRow=171&Paged=TRUE&PagedPrev=TRUE&View=%7BECDECFC9-707B-4F26-9830-F3EA40503071%7D&p_ID=300&p_SAMAFilePublishDate=20251222+21%3A00%3A00&p_SortBehavior=0",
    "https://www.sama.gov.sa/en-us/statistics/indices/pages/pos.aspx?PageFirstRow=151&Paged=TRUE&View=%7BECDECFC9-707B-4F26-9830-F3EA40503071%7D&p_ID=302&p_SAMAFilePublishDate=20260105+21%3A00%3A00&p_SortBehavior=0",
    "https://www.sama.gov.sa/en-us/indices/pages/pos.aspx?p_id=280&p_modified=20250805+13%3A12%3A08&p_samafilepublishdate=20250804+21%3A00%3A00&p_sortbehavior=0&paged=true&pagedprev=true&pagefirstrow=2761&view=%7Bcfcb1f9f-49c7-4bcc-8554-e968b1bb63aa%7D",
    "https://www.sama.gov.sa/en-us/indices/pages/pos.aspx?p_id=284&p_samafilepublishdate=20250901+21%3A00%3A00&p_sortbehavior=0&paged=true&pagedprev=true&view=%7Bcfcb1f9f-49c7-4bcc-8554-e968b1bb63aa%7D",
    "https://www.sama.gov.sa/en-us/indices/pages/pos.aspx?p_id=288&p_samafilepublishdate=20250929+21%3A00%3A00&p_sortbehavior=0&paged=true&pagedprev=true&view=%7Bcfcb1f9f-49c7-4bcc-8554-e968b1bb63aa%7D",
    "https://www.sama.gov.sa/en-us/indices/pages/pos.aspx?p_id=292&p_samafilepublishdate=20251027+21%3A00%3A00&p_sortbehavior=0&paged=true&pagedprev=true&view=%7Bcfcb1f9f-49c7-4bcc-8554-e968b1bb63aa%7D",
    "https://www.sama.gov.sa/en-us/indices/pages/pos.aspx?p_id=296&p_samafilepublishdate=20251124+21%3A00%3A00&p_sortbehavior=0&paged=true&pagedprev=true&view=%7Bcfcb1f9f-49c7-4bcc-8554-e968b1bb63aa%7D",
]

PDF_STRING_RE = re.compile(r'''(?:\\u002f|/)[^\"']+?\.pdf|(?:Weekly|POS)[_%A-Za-z0-9\-(). ]+\.pdf''', re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(raw: str) -> str:
    raw = unquote(raw).replace("&amp;", "&")
    raw = raw.replace("\\u002f", "/").replace("\\/", "/")
    if raw.startswith("http"):
        return raw
    if raw.startswith("/"):
        return urljoin(BASE, raw)
    return urljoin(BASE + "/", raw)


def extract_refs(html: str) -> set[str]:
    refs = set()
    for m in re.finditer(r'''FileRef\\?\"\s*:\s*\\?\"([^\"]+?\.pdf)''', html, re.I):
        refs.add(m.group(1))
    for m in PDF_STRING_RE.finditer(html):
        refs.add(m.group(0))
    return refs


def main():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 Sales-Sentinel academic data verifier"})
    refs=set(); page_stats=[]
    for url in ANCHORS:
        try:
            r=s.get(url,timeout=45)
            if r.status_code != 200:
                page_stats.append({"url":url,"final_url":r.url,"status":r.status_code,"pdf_strings_found":0,"bytes":len(r.content),"accepted":False})
                continue
            found=extract_refs(r.text)
            refs |= found
            page_stats.append({"url":url,"final_url":r.url,"status":r.status_code,"pdf_strings_found":len(found),"bytes":len(r.content),"accepted":True})
        except Exception as exc:
            page_stats.append({"url":url,"status":None,"pdf_strings_found":0,"accepted":False,"error":repr(exc)})

    candidates=sorted({normalize(x) for x in refs if '.pdf' in x.lower()})
    candidates=[u for u in candidates if '/indices/' in u.lower() or '/statistics/indices/' in u.lower()]

    manifest={"source":"Saudi Central Bank (SAMA) weekly POS archive","cutoff":CUTOFF,"anchors":ANCHORS,"page_stats":page_stats,"candidate_urls":candidates,"downloads":[]}
    seen_names=set()
    for url in candidates:
        name=unquote(url.split('?')[0].split('/')[-1])
        if not any(y in name for y in ('2025','2026')) or name in seen_names:
            continue
        try:
            r=s.get(url,timeout=60,allow_redirects=True)
            row={"url":url,"name":name,"status":r.status_code,"content_type":r.headers.get('content-type'),"final_url":r.url,"bytes":len(r.content)}
            if r.status_code==200 and r.content.startswith(b'%PDF'):
                path=OUT/name
                path.write_bytes(r.content)
                row.update({"is_pdf":True,"sha256":sha256(path)})
                seen_names.add(name)
            else:
                row['is_pdf']=False
            manifest['downloads'].append(row)
        except Exception as exc:
            manifest['downloads'].append({"url":url,"name":name,"error":repr(exc)})

    manifest['downloaded_pdfs']=sum(1 for x in manifest['downloads'] if x.get('is_pdf'))
    manifest['downloaded_names']=sorted({x['name'] for x in manifest['downloads'] if x.get('is_pdf')})
    manifest['successful_anchor_pages']=sum(1 for x in page_stats if x.get('accepted'))
    manifest['failed_anchor_pages']=sum(1 for x in page_stats if not x.get('accepted'))
    (REPORT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({"anchor_pages":len(ANCHORS),"successful_anchors":manifest['successful_anchor_pages'],"failed_anchors":manifest['failed_anchor_pages'],"candidate_urls":len(candidates),"downloaded_pdfs":manifest['downloaded_pdfs'],"names":manifest['downloaded_names']},indent=2))
    if manifest['downloaded_pdfs'] < 20:
        raise RuntimeError(f"Expected at least 20 distinct recent SAMA PDFs; got {manifest['downloaded_pdfs']}")


if __name__=='__main__':
    main()
