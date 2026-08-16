from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

from app.database import create_all, init_engine, session_scope
from app.models import ImportJob
from app.services.bootstrap import ensure_runtime_schema
from app.services.decline_explainer import explain_decline_drivers
from app.services.sales_importer import ingest_csv, inspect_csv


def test_decline_explainer_ranks_data_supported_signals(tmp_path: Path):
    db_path = tmp_path / "drivers.sqlite3"
    init_engine(f"sqlite:///{db_path}")
    create_all()
    ensure_runtime_schema()

    csv_path = tmp_path / "decline.csv"
    fields = [
        "TRX DATE", "TRX NUMBER", "SALES CHANNEL", "CUSTOMER NUMBER",
        "ITEM CODE", "QUANTITY", "Unit Price", "Net Amount", "TOTAL AMOUNT",
    ]
    start = date(2026, 1, 1)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(14):
            recent = i >= 7
            qty = 2 if recent else 5
            net = 200 if recent else 500
            writer.writerow({
                "TRX DATE": (start + timedelta(days=i)).isoformat(),
                "TRX NUMBER": f"T-{i:02d}",
                "SALES CHANNEL": "Online",
                "CUSTOMER NUMBER": f"C-{i % 3}",
                "ITEM CODE": "SKU-1",
                "QUANTITY": qty,
                "Unit Price": 100,
                "Net Amount": net,
                "TOTAL AMOUNT": net * 1.15,
            })

    mode, total, accepted, errors = inspect_csv(csv_path)
    assert (mode, total, accepted, errors) == ("redsea", 14, 14, [])

    with session_scope() as db:
        job = ImportJob(
            filename=csv_path.name,
            file_sha256="e" * 64,
            status="importing",
            total_rows=14,
            accepted_rows=0,
            rejected_rows=0,
        )
        db.add(job)
        db.flush()
        result = ingest_csv(db, csv_path, job.id, mode)
        assert result["inserted_rows"] == 14

    with session_scope() as db:
        explanation = explain_decline_drivers(db)

    assert explanation["available"] is True
    assert explanation["causal_claim"] is False
    assert explanation["sales_change_pct"] == -60.0
    codes = {item["code"] for item in explanation["drivers"]}
    assert "quantity_decline" in codes
    assert "basket_decline" in codes
    assert "channel_decline" in codes
    assert abs(sum(item["strength_pct"] for item in explanation["drivers"]) - 100.0) <= 0.2
