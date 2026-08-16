from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import Alert, Branch, ImportJob, ModelRun, Product, Sale
from app.services.dashboard_service import dashboard_summary
from app.services.data_scope import preferred_sales_condition


def test_dashboard_switches_from_seed_to_imported_sales_only(tmp_path: Path):
    database_path = tmp_path / "scope.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "scope-test-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    create_app(TestConfig)

    with session_scope() as db:
        condition, mode, imported_count = preferred_sales_condition(db)
        assert mode == "seed"
        assert imported_count == 0
        seed_count = int(db.scalar(select(func.count(Sale.id)).where(condition)) or 0)
        assert seed_count > 0

        branch = db.scalar(select(Branch).order_by(Branch.id))
        product = db.scalar(select(Product).order_by(Product.id))
        assert branch is not None and product is not None

        # Simulate an old completed forecast/alert that belongs to the previous
        # seed history. It must disappear from the dashboard once newer imported
        # sales become the preferred data scope.
        db.add(ModelRun(
            model_name="legacy-seed",
            model_version="seed-v1",
            status="completed",
            horizon_days=7,
            filters_json={"data_mode": "seed_aggregate"},
            metrics_json={},
            data_start=date(2011, 1, 1),
            data_end=date(2011, 1, 30),
            sample_size=30,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        ))
        db.add(Alert(
            forecast_id=None,
            severity="medium",
            title_ar="تنبيه قديم",
            title_en="Old alert",
            message_ar="تنبيه يخص بيانات سابقة",
            message_en="Alert for previous data",
            is_read=False,
            is_resolved=False,
        ))

        job = ImportJob(
            filename="merchant.csv",
            file_sha256="c" * 64,
            status="imported",
            total_rows=2,
            accepted_rows=2,
            rejected_rows=0,
        )
        db.add(job)
        db.flush()
        for i, amount in enumerate((Decimal("500"), Decimal("700"))):
            db.add(Sale(
                sale_date=date(2026, 8, 10 + i),
                transaction_number=f"USER-{i}",
                transaction_type="INV",
                branch_id=branch.id,
                product_id=product.id,
                channel="Store",
                quantity=1,
                unit_price=amount,
                discount_amount=Decimal("0"),
                discount_percent=0.0,
                gross_sales=amount,
                net_sales=amount,
                vat_amount=Decimal("0"),
                total_amount=amount,
                inventory_available=False,
                is_promotion=False,
                seasonal_factor=1.0,
                is_demo=False,
                source_row_hash=hashlib.sha256(f"scope:{i}".encode()).hexdigest(),
                source_import_id=job.id,
            ))
        db.flush()

        condition, mode, imported_count = preferred_sales_condition(db)
        assert mode == "imported"
        assert imported_count == 2
        assert int(db.scalar(select(func.count(Sale.id)).where(condition)) or 0) == 2

        summary = dashboard_summary(db, None)
        assert summary["data_mode"] == "imported"
        assert summary["total_records"] == 2
        assert summary["data_end"] == date(2026, 8, 11)
        assert [point["date"] for point in summary["series"]] == ["2026-08-10", "2026-08-11"]
        assert summary["current_sales"] == 1200.0
        assert summary["latest_run"] is None
        assert summary["forecast_series"] == []
        assert summary["active_alerts"] == 0
        assert summary["top_alerts"] == []
