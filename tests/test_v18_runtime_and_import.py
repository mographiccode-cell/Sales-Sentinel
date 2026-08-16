from __future__ import annotations

import csv
import json
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import inspect, text

from app.database import create_all, init_engine, session_scope
from app.models import ImportJob
from app.services.bootstrap import ensure_runtime_schema
from app.services import portable_decline_engine as v18
from app.services.sales_importer import _date, ingest_csv, inspect_csv


def test_v18_artifact_and_pure_runtime_score():
    artifact = v18.load_artifact()
    assert len(artifact["feature_names"]) == 96
    assert len(artifact["trees"]) == 1000
    start = date(2023, 1, 1)
    daily = []
    for i in range(70):
        daily.append({
            "date": start + timedelta(days=i),
            "sama_calibrated_net_sales_sar": 10000.0 + (i % 7) * 250.0,
            "gross_sales_sar": 10500.0 + (i % 7) * 250.0,
            "invoice_count": 40.0 + (i % 5),
            "unique_observed_customers": 30.0 + (i % 4),
            "unique_products": 20.0 + (i % 3),
            "units": 60.0 + (i % 6),
            "average_invoice_value_sar": 250.0,
            "return_rate_value": 0.02,
            "transaction_rows": 50.0 + (i % 5),
        })
    features, _ = v18._feature_map(daily)
    row = v18._prepare_row(artifact, features)
    score = v18._score(artifact, row)
    assert len(row) == 96
    assert 0.0 <= score <= 1.0


def test_model_registry_matches_runtime_artifact():
    registry_path = Path(__file__).resolve().parents[1] / "models" / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    runtime = registry["models"]["portable_runtime"]
    development = registry["models"]["merchant_decline_development_best"]
    point = registry["models"]["point_sales_forecast"]
    artifact = v18.load_artifact()

    assert development["version"] == "V13.1"
    assert development["status"] == "DEVELOPMENT_BEST"
    assert development["metrics"]["recall"] == 0.8254
    assert development["metrics"]["precision"] == 0.4444

    assert runtime["version"] == artifact["version"]
    assert runtime["artifact"] == "models/sales_sentinel_portable_v18.json.gz"
    assert runtime["feature_count"] == len(artifact["feature_names"]) == 96
    assert runtime["tree_count"] == len(artifact["trees"]) == 1000
    assert runtime["history_required_days"] == int(artifact.get("history_required_days", 56))
    assert runtime["red_severity_supported"] is False

    assert point["name"] == "ridge_raw_1"
    assert point["status"] == "LEGACY_POINT_FORECAST_ONLY"
    assert "decline" in point["must_not_be_used_for"].lower()


def test_uci_ambiguous_date_is_month_first():
    assert _date("12/1/2010 8:26", uci=True) == date(2010, 12, 1)
    assert _date("2023-07-04") == date(2023, 7, 4)


def test_schema_upgrade_and_idempotent_transaction_import(tmp_path: Path):
    db_path = tmp_path / "sales.sqlite3"
    engine = init_engine(f"sqlite:///{db_path}")
    create_all()
    ensure_runtime_schema()
    assert "customer_key" in {column["name"] for column in inspect(engine).get_columns("sales")}

    csv_path = tmp_path / "uci.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"])
        writer.writeheader()
        writer.writerow({"InvoiceNo": "536365", "StockCode": "85123A", "Description": "ITEM A", "Quantity": "2", "InvoiceDate": "12/1/2010 8:26", "UnitPrice": "10.00", "CustomerID": "17850", "Country": "United Kingdom"})
        writer.writerow({"InvoiceNo": "C536366", "StockCode": "85123A", "Description": "ITEM A", "Quantity": "-1", "InvoiceDate": "12/2/2010 9:00", "UnitPrice": "10.00", "CustomerID": "17850", "Country": "United Kingdom"})

    mode, total, accepted, errors = inspect_csv(csv_path)
    assert (mode, total, accepted, errors) == ("uci", 2, 2, [])

    with session_scope() as db:
        job = ImportJob(filename="uci.csv", file_sha256="0" * 64, status="importing", total_rows=2, accepted_rows=0, rejected_rows=0)
        db.add(job); db.flush()
        result1 = ingest_csv(db, csv_path, job.id, mode)
        assert result1["inserted_rows"] == 2

    with session_scope() as db:
        job2 = ImportJob(filename="uci.csv", file_sha256="1" * 64, status="importing", total_rows=2, accepted_rows=0, rejected_rows=0)
        db.add(job2); db.flush()
        result2 = ingest_csv(db, csv_path, job2.id, mode)
        assert result2["inserted_rows"] == 0
        rows = db.execute(text("SELECT sale_date, transaction_type, customer_key, net_sales FROM sales ORDER BY sale_date")).all()
        assert len(rows) == 2
        assert str(rows[0][0])[:10] == "2010-12-01"
        assert rows[0][2] == "17850"
        assert float(rows[0][3]) == 20.0
        assert rows[1][1] == "RETURN"
        assert float(rows[1][3]) == -10.0


def test_rich_import_activates_v18_end_to_end(tmp_path: Path):
    """Prove the production path: CSV -> SQLite -> 56-day features -> V18 trees."""
    db_path = tmp_path / "v18-e2e.sqlite3"
    init_engine(f"sqlite:///{db_path}")
    create_all()
    ensure_runtime_schema()

    csv_path = tmp_path / "rich_uci.csv"
    fields = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
    start = date(2023, 1, 1)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(70):
            day = start + timedelta(days=i)
            # Two customers and two products every calendar day make customer/product
            # observability explicit while sales trend changes through time.
            for j in range(2):
                writer.writerow({
                    "InvoiceNo": f"INV-{i:03d}-{j}",
                    "StockCode": f"SKU-{j}",
                    "Description": f"Product {j}",
                    "Quantity": str(1 + ((i + j) % 3)),
                    "InvoiceDate": day.isoformat(),
                    "UnitPrice": str(80 + (i % 7) * 4 + j * 15),
                    "CustomerID": f"CUST-{j}",
                    "Country": "Saudi Arabia",
                })

    mode, total, accepted, errors = inspect_csv(csv_path)
    assert mode == "uci"
    assert total == accepted == 140
    assert errors == []
    with session_scope() as db:
        job = ImportJob(filename=csv_path.name, file_sha256="2" * 64, status="importing", total_rows=total, accepted_rows=0, rejected_rows=0)
        db.add(job); db.flush()
        result = ingest_csv(db, csv_path, job.id, mode)
        assert result["inserted_rows"] == 140

    with session_scope() as db:
        risk = v18.assess_decline_risk(db)
        assert risk["available"] is True
        assert risk["model_version"] == "SALES-SENTINEL-V18-PORTABLE-EXTRATREES-RUNTIME"
        assert risk["feature_count"] == 96
        assert risk["tree_count"] == 1000
        assert risk["history_days"] == 70
        assert risk["policy_mode"] == "static"
        assert 0.0 <= risk["score"] <= 1.0
        assert isinstance(risk["alert"], bool)
