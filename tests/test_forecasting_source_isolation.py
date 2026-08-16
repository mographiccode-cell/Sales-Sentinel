from __future__ import annotations

import hashlib
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.forecasting.routes import _daily_sales_rows
from app.models import Branch, Product, Sale


def test_user_history_never_mixes_with_seed_aggregates(tmp_path: Path):
    database_path = tmp_path / "forecast-source.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "forecast-source-test"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    create_app(TestConfig)

    with session_scope() as db:
        seed_rows, seed_mode = _daily_sales_rows(db)
        assert seed_mode == "seed_aggregate"
        assert len(seed_rows) >= 28

        branch = db.scalar(select(Branch).order_by(Branch.id))
        product = db.scalar(select(Product).order_by(Product.id))
        assert branch is not None and product is not None

        # Add only ten days of rich user transactions. The correct behavior is
        # to expose those ten days and let the forecasting route reject them as
        # insufficient, never combine them with the older seeded UCI history.
        start = date(2026, 1, 1)
        for i in range(10):
            day = start + timedelta(days=i)
            db.add(Sale(
                sale_date=day,
                transaction_number=f"RICH-{i}",
                transaction_type="INV",
                branch_id=branch.id,
                product_id=product.id,
                channel="Store",
                quantity=1,
                unit_price=Decimal("100"),
                discount_amount=Decimal("0"),
                discount_percent=0.0,
                gross_sales=Decimal("100"),
                net_sales=Decimal("100"),
                vat_amount=Decimal("15"),
                total_amount=Decimal("115"),
                inventory_available=False,
                is_promotion=False,
                seasonal_factor=1.0,
                is_demo=False,
                source_row_hash=hashlib.sha256(f"rich:{i}".encode()).hexdigest(),
            ))
        db.flush()

        rich_rows, rich_mode = _daily_sales_rows(db)
        assert rich_mode == "transaction_level_insufficient"
        assert len(rich_rows) == 10
        assert rich_rows[0][0] == start
        assert rich_rows[-1][0] == start + timedelta(days=9)

        # A complete explicit daily import may be used as a coherent fallback,
        # but it must remain isolated from both rich rows and seeded aggregates.
        daily_start = date(2026, 3, 1)
        for i in range(30):
            day = daily_start + timedelta(days=i)
            db.add(Sale(
                sale_date=day,
                transaction_number=f"DAILY-{i}",
                transaction_type="DAILY_IMPORT",
                branch_id=branch.id,
                product_id=product.id,
                channel="Aggregate",
                quantity=0,
                unit_price=Decimal("1"),
                discount_amount=Decimal("0"),
                discount_percent=0.0,
                gross_sales=Decimal("1000"),
                net_sales=Decimal("1000"),
                vat_amount=Decimal("0"),
                total_amount=Decimal("1000"),
                inventory_available=False,
                is_promotion=False,
                seasonal_factor=1.0,
                is_demo=False,
                source_row_hash=hashlib.sha256(f"daily:{i}".encode()).hexdigest(),
            ))
        db.flush()

        daily_rows, daily_mode = _daily_sales_rows(db)
        assert daily_mode == "daily_import"
        assert len(daily_rows) == 30
        assert daily_rows[0][0] == daily_start
        assert daily_rows[-1][0] == daily_start + timedelta(days=29)
