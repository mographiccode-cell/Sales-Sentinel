from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest
from sqlalchemy import inspect, text

from app.database import Base, create_all, get_engine, init_engine, normalize_database_url, session_scope
from app.models import ImportJob
from app.services.bootstrap import ensure_runtime_schema
from app.services.sales_importer import ingest_csv, inspect_csv


def test_common_postgres_urls_are_normalized_to_psycopg():
    assert normalize_database_url("postgres://u:p@host/db") == "postgresql+psycopg://u:p@host/db"
    assert normalize_database_url("postgresql://u:p@host/db") == "postgresql+psycopg://u:p@host/db"
    assert normalize_database_url("postgresql+psycopg://u:p@host/db") == "postgresql+psycopg://u:p@host/db"
    assert normalize_database_url("sqlite:///local.db") == "sqlite:///local.db"


def test_postgres_schema_and_duplicate_safe_transaction_ingestion(tmp_path: Path):
    url = os.getenv("POSTGRES_TEST_URL")
    if not url:
        pytest.skip("POSTGRES_TEST_URL is not configured")

    engine = init_engine(url)
    Base.metadata.drop_all(bind=engine)
    create_all()
    ensure_runtime_schema()

    assert get_engine().dialect.name == "postgresql"
    assert "customer_key" in {column["name"] for column in inspect(engine).get_columns("sales")}

    csv_path = tmp_path / "postgres_uci.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
        )
        writer.writeheader()
        writer.writerow({
            "InvoiceNo": "PG-001", "StockCode": "SKU-PG", "Description": "Postgres item",
            "Quantity": "2", "InvoiceDate": "2023-07-01", "UnitPrice": "10.00",
            "CustomerID": "PG-CUST", "Country": "Saudi Arabia",
        })
        writer.writerow({
            "InvoiceNo": "PG-002", "StockCode": "SKU-PG", "Description": "Postgres item",
            "Quantity": "1", "InvoiceDate": "2023-07-02", "UnitPrice": "12.00",
            "CustomerID": "PG-CUST", "Country": "Saudi Arabia",
        })

    mode, total, accepted, errors = inspect_csv(csv_path)
    assert (mode, total, accepted, errors) == ("uci", 2, 2, [])

    with session_scope() as db:
        job = ImportJob(
            filename=csv_path.name,
            file_sha256="a" * 64,
            status="importing",
            total_rows=2,
            accepted_rows=0,
            rejected_rows=0,
        )
        db.add(job)
        db.flush()
        first = ingest_csv(db, csv_path, job.id, mode)
        assert first["inserted_rows"] == 2
        assert first["duplicate_rows"] == 0

    with session_scope() as db:
        job = ImportJob(
            filename=csv_path.name,
            file_sha256="b" * 64,
            status="importing",
            total_rows=2,
            accepted_rows=0,
            rejected_rows=0,
        )
        db.add(job)
        db.flush()
        second = ingest_csv(db, csv_path, job.id, mode)
        assert second["inserted_rows"] == 0
        assert second["duplicate_rows"] == 2
        rows = db.execute(text("SELECT transaction_number, customer_key, CAST(net_sales AS DOUBLE PRECISION) FROM sales ORDER BY transaction_number")).all()
        assert rows == [("PG-001", "PG-CUST", 20.0), ("PG-002", "PG-CUST", 12.0)]
