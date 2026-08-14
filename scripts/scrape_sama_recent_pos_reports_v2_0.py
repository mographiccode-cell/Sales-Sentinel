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

PDF_RE = re.compile(r'''(?:https?://[^\"'<> ]+|/[^\"'<> ]+)?Weekly[_%A-Za-z0-9\-(). ]+\.pdf''', re.I)
DATE_RE = re.compile(r"(20\d{2})[-_/ ]?(\d{1,2})[-_/ ]?(\d{1,2})")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_candidate(raw: str) -> str:
    raw = unquote(raw).replace("&amp;", "&")
    if raw.startswith("http"):
        return raw
    return urljoin(BASE, raw)


def extract_pdf_urls(html: str) -> set[str]:
    urls = set()
    # href/src-like quoted values
    for m in re.finditer(r'''[\"']([^\"']+\.pdf(?:\?[^\"']*)?)[\"']''', html, re.I):
        urls.add(normalize_candidate(m.group(1)))
    # SharePoint/JS escaped snippets containing the published filename
    for m in PDF_RE.finditer(html):
        urls.add(normalize_candidate(m.group(0)))
    return {u for u in urls if "Weekly" in u and ".pdf" in u.lower()}


def main():
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 Sales-Sentinel academic data verifier"})
    discovered = set()
    page_stats = []

    # SAMA SharePoint view exposes 10 rows/page. Scan a broad window and dedupe URLs.
    for first in range(1, 421, 10):
        params = {"PageFirstRow": first, "Paged": "TRUE"}
        r = s.get(LIST, params=params, timeout=45)
        r.raise_for_status()
        urls = extract_pdf_urls(r.text)
        discovered |= urls
        page_stats.append({"first_row": first, "status": r.status_code, "pdf_urls_found": len(urls), "bytes": len(r.content)})

    # If SAMA serves the view differently, preserve HTML diagnostics for reproducibility.
    manifest = {
        "source": LIST,
        "cutoff": CUTOFF,
        "pages_scanned": len(page_stats),
        "discovered_pdf_urls": len(discovered),
        "page_stats": page_stats,
        "downloads": [],
    }

    for url in sorted(discovered):
        name = unquote(url.split("?")[0].split("/")[-1])
        # filenames include report week; download all discovered recent-looking files and filter later by parsed report content.
        if not any(y in name for y in ("2025", "2026")):
            continue
        try:
            r = s.get(url, timeout=60)
            if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                manifest["downloads"].append({"url": url, "name": name, "status": r.status_code, "is_pdf": False, "bytes": len(r.content)})
                continue
            path = OUT / name
            path.write_bytes(r.content)
            manifest["downloads"].append({"url": url, "name": name, "status": 200, "is_pdf": True, "bytes": len(r.content), "sha256": sha256(path)})
        except Exception as exc:
            manifest["downloads"].append({"url": url, "name": name, "error": repr(exc)})

    manifest["downloaded_pdfs"] = sum(1 for x in manifest["downloads"] if x.get("is_pdf"))
    (REPORT / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"pages_scanned": manifest["pages_scanned"], "discovered": manifest["discovered_pdf_urls"], "downloaded_pdfs": manifest["downloaded_pdfs"]}, indent=2))
    if manifest["downloaded_pdfs"] < 10:
        raise RuntimeError(f"Too few recent SAMA PDFs downloaded: {manifest['downloaded_pdfs']}")


if __name__ == "__main__":
    main()
