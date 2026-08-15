from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(os.environ.get("SAUDI_KAGGLE_DIR", "/tmp/saudi_kaggle"))
OUT = ROOT / "reports" / "external_saudi_kaggle_inspection"
OUT.mkdir(parents=True, exist_ok=True)
REPORT = OUT / "inspection_report.json"
SUMMARY = OUT / "inspection_summary.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_tabular(path: Path) -> dict:
    info = {"name": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    try:
        if path.suffix.lower() == ".csv":
            d = pd.read_csv(path, low_memory=False)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            d = pd.read_excel(path)
        elif path.suffix.lower() == ".parquet":
            d = pd.read_parquet(path)
        else:
            info["status"] = "unsupported"
            return info
        info["status"] = "ok"
        info["rows"] = int(len(d)); info["columns"] = int(len(d.columns)); info["column_names"] = [str(c) for c in d.columns]
        info["duplicate_rows"] = int(d.duplicated().sum())
        info["missing_cells"] = int(d.isna().sum().sum())

        date_candidates = [c for c in d.columns if any(k in str(c).lower() for k in ["date", "time", "timestamp", "invoice date"])]
        info["date_candidates"] = [str(c) for c in date_candidates]
        ranges = {}
        for c in date_candidates[:12]:
            q = pd.to_datetime(d[c], errors="coerce")
            if q.notna().sum() >= max(3, int(.2 * len(d))):
                ranges[str(c)] = {"min": str(q.min()), "max": str(q.max()), "unique": int(q.nunique()), "parsed": int(q.notna().sum())}
        info["date_ranges"] = ranges

        numeric = d.select_dtypes(include=[np.number])
        stats = {}
        for c in numeric.columns[:80]:
            x = pd.to_numeric(numeric[c], errors="coerce")
            if x.notna().any():
                stats[str(c)] = {"min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "unique": int(x.nunique())}
        info["numeric_summary"] = stats

        keys = ["transaction", "invoice", "customer", "product", "category", "price", "quantity", "sales", "total", "amount", "city", "branch", "store", "date"]
        info["semantic_columns"] = {k: [str(c) for c in d.columns if k in str(c).lower()] for k in keys}
        info["sample_rows"] = d.head(5).astype(str).to_dict(orient="records")
    except Exception as e:
        info["status"] = "error"; info["error"] = repr(e)
    return info


def main():
    files = sorted([p for p in SRC.rglob("*") if p.is_file()])
    inspected = []
    for p in files:
        if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet"}:
            inspected.append(inspect_tabular(p))
        else:
            inspected.append({"name": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p), "status": "non_tabular"})

    usable = [x for x in inspected if x.get("status") == "ok"]
    total_rows = sum(x.get("rows", 0) for x in usable)
    has_date = any(bool(x.get("date_ranges")) for x in usable)
    txn_like = any(any(x.get("semantic_columns", {}).get(k, []) for k in ["transaction", "invoice", "sales", "total", "amount"]) for x in usable)
    report = {
        "source": "Kaggle moha684/saudi-arabia-supermarket-data-expenses",
        "downloaded_file_count": len(files),
        "tabular_file_count": len(usable),
        "total_tabular_rows": int(total_rows),
        "has_parseable_date": bool(has_date),
        "has_transaction_or_sales_fields": bool(txn_like),
        "files": inspected,
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# External Saudi Kaggle Dataset Inspection", "",
        "- Source: **Kaggle moha684/saudi-arabia-supermarket-data-expenses**",
        f"- Files downloaded: **{len(files)}**",
        f"- Usable tabular files: **{len(usable)}**",
        f"- Total tabular rows: **{total_rows:,}**",
        f"- Parseable date present: **{has_date}**",
        f"- Transaction/sales-like fields present: **{txn_like}**", "",
    ]
    for x in usable:
        lines += [
            f"## {x['name']}",
            f"- Rows / columns: **{x['rows']:,} / {x['columns']}**",
            f"- SHA-256: `{x['sha256']}`",
            f"- Duplicate rows: **{x['duplicate_rows']:,}**",
            f"- Missing cells: **{x['missing_cells']:,}**",
            f"- Columns: {x['column_names']}",
            f"- Date ranges: {x['date_ranges']}",
            f"- Semantic columns: {x['semantic_columns']}", "",
        ]
    SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
