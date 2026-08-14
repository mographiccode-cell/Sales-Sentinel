from __future__ import annotations

import json
import re
from pathlib import Path

import parse_sama_recent_pos_reports_v2_0 as common

ROOT=Path(__file__).resolve().parents[1]
PDF_DIR=ROOT/'artifacts'/'sama_recent_v2_0'
OUT=ROOT/'reports'/'sama_city_v2_1'; OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/'pdf_layout_diagnostics.json'
TOKENS=['riyadh','jeddah','makkah','mecca','madina','madinah','dammam','khobar','abha','hail','buraidah','buraydah','tabuk','tabouk','other']


def contexts(lines, idx, radius=4):
    return [{'line_no':j+1,'text':lines[j]} for j in range(max(0,idx-radius),min(len(lines),idx+radius+1))]


def inspect(path:Path):
    text=common.pdf_text(path); lines=text.splitlines()
    markers=[(i,l) for i,l in enumerate(lines) if 'Table 2.1' in l]
    token_hits=[]
    for i,line in enumerate(lines):
        low=common.norm(line)
        if any(t in low for t in TOKENS):
            token_hits.append({'matched_line':i+1,'normalized':low,'context':contexts(lines,i,4)})
    # Capture chunks around every Table 2.1 marker, not assuming the city table starts after the literal marker.
    marker_chunks=[]
    for i,l in markers:
        marker_chunks.append({'marker_line':i+1,'marker':l,'chunk':[{'line_no':j+1,'text':lines[j]} for j in range(max(0,i-20),min(len(lines),i+180))]})
    return {
        'file':path.name,
        'total_text_lines':len(lines),
        'table_2_1_markers':len(markers),
        'marker_chunks':marker_chunks,
        'city_token_hits':token_hits[:100],
    }


def main():
    paths=sorted(PDF_DIR.glob('*.pdf'))
    if not paths: raise RuntimeError('No downloaded SAMA PDFs found')
    preferred=[]
    for needle in ['26_Jul_2025','26-Jul-2025','26_Jul','8th_Aug_2026','1st_Aug_2026','02_aug_2025']:
        m=[p for p in paths if needle.lower() in p.name.lower()]
        if m and m[0] not in preferred: preferred.append(m[0])
    # Ensure at least 4 representative PDFs spanning taxonomy versions.
    for p in [paths[0],paths[len(paths)//2],paths[-1]]:
        if p not in preferred: preferred.append(p)
    selected=preferred[:5]
    report={'pdf_count':len(paths),'selected':[p.name for p in selected],'documents':[inspect(p) for p in selected]}
    REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({
        'pdf_count':len(paths),
        'selected':report['selected'],
        'summary':[{
            'file':d['file'],'lines':d['total_text_lines'],'table_2_1_markers':d['table_2_1_markers'],
            'city_token_hits':len(d['city_token_hits']),
            'first_hits':d['city_token_hits'][:8],
        } for d in report['documents']]
    },indent=2))

if __name__=='__main__': main()
