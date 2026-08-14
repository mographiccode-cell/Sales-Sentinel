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
LIST = "https://www.sama.gov.sa/en-US/Statistics/Indices/pages/pos.aspx"
CUTOFF = "2025-07-07"

PDF_RE = re.compile(r'''(?:https?://[^\"'<> ]+|/[^\"'<> ]+)?(?:Weekly|POS)[_%A-Za-z0-9\-(). ]+\.pdf''', re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_candidate(raw: str) -> str:
    raw = unquote(raw).replace("&amp;", "&").replace("\\u002f", "/")
    if raw.startswith("http"):
        return raw
    return urljoin(BASE, raw)


def extract_pdf_strings(html: str) -> set[str]:
    values = set()
    # quoted attributes or script strings containing .pdf
    for m in re.finditer(r'''[\"']([^\"']+\.pdf(?:\?[^\"']*)?)[\"']''', html, re.I):
        values.add(m.group(1))
    for m in PDF_RE.finditer(html):
        values.add(m.group(0))
    return values


def contexts(html: str, limit=20):
    out=[]
    for m in re.finditer(r'[^<>\"\']+\.pdf', html, re.I):
        a=max(0,m.start()-500); b=min(len(html),m.end()+500)
        text=html[a:b].replace('\n',' ').replace('\r',' ')
        out.append(text)
        if len(out)>=limit: break
    return out


def main():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 Sales-Sentinel academic data verifier"})
    raw_strings=set(); page_stats=[]; html_contexts=[]

    # Use the SharePoint pagination parameters known to be accepted by SAMA.
    for first in range(1, 421, 10):
        params={"PageFirstRow": first, "Paged": "TRUE"}
        r=s.get(LIST, params=params, timeout=45)
        r.raise_for_status()
        found=extract_pdf_strings(r.text)
        raw_strings |= found
        if found and len(html_contexts)<40:
            html_contexts.extend(contexts(r.text, limit=10))
        page_stats.append({"first_row":first,"status":r.status_code,"pdf_strings_found":len(found),"bytes":len(r.content),"final_url":r.url})

    candidates=sorted({normalize_candidate(x) for x in raw_strings})
    manifest={
        "source":LIST,"cutoff":CUTOFF,"pages_scanned":len(page_stats),
        "raw_pdf_strings":sorted(raw_strings),"candidate_urls":candidates,
        "page_stats":page_stats,"downloads":[]
    }
    (REPORT/'html_contexts.json').write_text(json.dumps(html_contexts,indent=2),encoding='utf-8')

    for url in candidates:
        name=unquote(url.split('?')[0].split('/')[-1])
        if not any(y in name for y in ('2025','2026')): continue
        try:
            r=s.get(url,timeout=60,allow_redirects=True)
            row={"url":url,"name":name,"status":r.status_code,"content_type":r.headers.get('content-type'),"final_url":r.url,"bytes":len(r.content),"prefix":r.content[:20].hex()}
            if r.status_code==200 and r.content.startswith(b'%PDF'):
                path=OUT/name; path.write_bytes(r.content)
                row.update({"is_pdf":True,"sha256":sha256(path)})
            else: row['is_pdf']=False
            manifest['downloads'].append(row)
        except Exception as exc:
            manifest['downloads'].append({"url":url,"name":name,"error":repr(exc)})

    manifest['downloaded_pdfs']=sum(1 for x in manifest['downloads'] if x.get('is_pdf'))
    (REPORT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({
        "pages_scanned":manifest['pages_scanned'],
        "raw_pdf_strings":len(raw_strings),
        "candidate_urls":candidates[:10],
        "downloaded_pdfs":manifest['downloaded_pdfs'],
        "download_attempts":manifest['downloads'][:10],
        "html_context_samples":html_contexts[:3],
    },indent=2))


if __name__=='__main__':
    main()
