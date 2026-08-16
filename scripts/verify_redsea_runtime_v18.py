from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from sqlalchemy import text

from app.database import create_all, init_engine, session_scope
from app.models import ImportJob
from app.services import portable_decline_engine as v18
from app.services.bootstrap import ensure_runtime_schema
from app.services.sales_importer import ingest_csv, inspect_csv
from app.services.tabular_upload import normalize_tabular_upload

EXPECTED_SHA256 = "dd994d25042e54119babe058820b02e00b3443fe18150b9213f9c6929466a645"
EXPECTED_SIZE = 354721
OUT_DIR = Path("reports/external_redsea_runtime_v18")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    source = Path(os.environ.get("REDSEA_XLSX", "/tmp/RedSea_Data_Cleaned.xlsx"))
    if not source.exists():
        raise SystemExit(f"Missing Redsea source: {source}")

    source_sha = sha256_file(source)
    source_size = source.stat().st_size
    if source_sha != EXPECTED_SHA256:
        raise SystemExit(f"Unexpected Redsea SHA-256: {source_sha}")
    if source_size != EXPECTED_SIZE:
        raise SystemExit(f"Unexpected Redsea file size: {source_size}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime_csv, generated = normalize_tabular_upload(source)
    try:
        mode, total, accepted, validation_errors = inspect_csv(runtime_csv)
        if mode != "redsea":
            raise SystemExit(f"Expected redsea mode, got {mode}")
        if accepted <= 0:
            raise SystemExit("Redsea validation accepted no rows")

        db_path = Path("/tmp/redsea_runtime_v18.sqlite3")
        db_path.unlink(missing_ok=True)
        init_engine(f"sqlite:///{db_path}")
        create_all()
        ensure_runtime_schema()

        with session_scope() as db:
            job = ImportJob(
                filename=source.name,
                file_sha256=source_sha,
                status="importing",
                total_rows=total,
                accepted_rows=0,
                rejected_rows=total - accepted,
                error_details={"mode": mode, "source_format": "xlsx"},
            )
            db.add(job)
            db.flush()
            ingestion = ingest_csv(db, runtime_csv, job.id, mode)
            job.status = "imported" if ingestion["inserted_rows"] else "imported_no_new_rows"
            job.accepted_rows = ingestion["inserted_rows"]
            job.rejected_rows = (total - accepted) + ingestion["rejected_rows"]

        with session_scope() as db:
            risk = v18.assess_decline_risk(db)
            stats = db.execute(text("""
                SELECT
                    COUNT(*) AS rows,
                    COUNT(DISTINCT sale_date) AS days,
                    COUNT(DISTINCT transaction_number) AS transactions,
                    COUNT(DISTINCT customer_key) AS customers,
                    COUNT(DISTINCT product_id) AS products,
                    MIN(sale_date) AS min_date,
                    MAX(sale_date) AS max_date,
                    SUM(CAST(net_sales AS REAL)) AS net_sales
                FROM sales
            """)).mappings().one()

        report = {
            "status": "PASS" if risk.get("available") else "FAIL",
            "purpose": "Runtime integration verification only; not an accuracy or blind-validation claim.",
            "source": {
                "doi": "10.17632/9c87bd42ct.1",
                "file": source.name,
                "sha256": source_sha,
                "bytes": source_size,
            },
            "validation": {
                "mode": mode,
                "total_rows": total,
                "accepted_rows": accepted,
                "rejected_rows": total - accepted,
                "errors": validation_errors,
            },
            "ingestion": ingestion,
            "database": {
                "rows": int(stats["rows"] or 0),
                "days": int(stats["days"] or 0),
                "transactions": int(stats["transactions"] or 0),
                "customers": int(stats["customers"] or 0),
                "products": int(stats["products"] or 0),
                "date_start": str(stats["min_date"]),
                "date_end": str(stats["max_date"]),
                "net_sales": float(stats["net_sales"] or 0.0),
            },
            "v18": {
                key: (value.isoformat() if hasattr(value, "isoformat") else value)
                for key, value in risk.items()
            },
            "scientific_boundary": [
                "No model or threshold was selected using this runtime test.",
                "This proves the real Saudi XLSX can flow through the deployed ingestion and V18 scoring path.",
                "It does not prove production predictive performance or replace fresh independent longitudinal validation.",
            ],
        }

        if not risk.get("available"):
            raise SystemExit(f"V18 unavailable on real Redsea runtime data: {risk}")
        if int(risk.get("feature_count", 0)) != 96 or int(risk.get("tree_count", 0)) != 1000:
            raise SystemExit("Unexpected V18 runtime contract")

        (OUT_DIR / "runtime_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = f"""# Redsea -> Sales Sentinel V18 Runtime Verification

- Status: **{report['status']}**
- DOI: **10.17632/9c87bd42ct.1**
- Source SHA-256: `{source_sha}`
- Source rows validated: **{total:,}**
- Rows inserted after duplicate protection: **{ingestion['inserted_rows']:,}**
- Duplicate rows ignored: **{ingestion['duplicate_rows']:,}**
- Calendar history: **{stats['days']} days** ({stats['min_date']} -> {stats['max_date']})
- Distinct transactions: **{stats['transactions']}**
- Distinct customers: **{stats['customers']}**
- Distinct products: **{stats['products']}**
- V18 available: **{risk.get('available')}**
- V18 version: **{risk.get('model_version')}**
- Feature count: **{risk.get('feature_count')}**
- Tree count: **{risk.get('tree_count')}**
- Policy mode: **{risk.get('policy_mode')}**
- Latest runtime risk score: **{float(risk.get('score', 0.0)):.6f}**
- Alert decision: **{risk.get('alert')}**
- RED supported: **{risk.get('red_supported')}**

Scientific boundary: this is a real-file runtime integration proof, not an accuracy claim and not fresh blind external validation.
"""
        (OUT_DIR / "runtime_summary.md").write_text(summary, encoding="utf-8")
        print(summary)
    finally:
        if generated:
            runtime_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
