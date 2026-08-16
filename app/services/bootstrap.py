from __future__ import annotations

import csv
import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, inspect, select, text

from app.database import create_all, get_engine, session_scope
from app.models import Branch, Category, Permission, Product, Region, Role, Sale, SystemHealth, SystemSetting, User
from app.services.security import hash_password

BASE_DIR = Path(__file__).resolve().parents[2]


def ensure_runtime_schema() -> None:
    """Apply small additive runtime-safe upgrades to existing SQLite databases.

    SQLAlchemy ``create_all`` creates new tables but does not add columns to an
    existing table. ``customer_key`` is deliberately kept as an additive SQL
    column so old databases remain readable while transaction imports can retain
    a real customer identifier for V18 unique-customer features.
    """
    engine = get_engine()
    if engine is None:
        raise RuntimeError("Database engine is not initialized")
    columns = {column["name"] for column in inspect(engine).get_columns("sales")}
    if "customer_key" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE sales ADD COLUMN customer_key VARCHAR(100)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_sales_customer_key ON sales (customer_key)"))


def ensure_seed_data() -> None:
    """Initialize SQLite from verified daily aggregates without inventing invoices."""
    create_all()
    ensure_runtime_schema()
    with session_scope() as db:
        if int(db.scalar(select(func.count(User.id))) or 0) > 0:
            return
        permissions = {}
        definitions = [
            ("dashboard.view", "عرض لوحة المعلومات", "View dashboard"),
            ("sales.view", "عرض المبيعات", "View sales"),
            ("forecasts.run", "تشغيل التوقعات", "Run forecasts"),
            ("imports.manage", "إدارة الاستيراد", "Manage imports"),
            ("reports.export", "تصدير التقارير", "Export reports"),
            ("users.manage", "إدارة المستخدمين", "Manage users"),
            ("system.manage", "إدارة النظام", "Manage system"),
            ("branches.view_all", "عرض جميع الفروع", "View all branches"),
        ]
        for code, ar, en in definitions:
            permission = Permission(code=code, name_ar=ar, name_en=en)
            permissions[code] = permission; db.add(permission)
        admin_role = Role(code="admin", name_ar="مسؤول النظام", name_en="System Administrator", permissions=list(permissions.values()))
        analyst_role = Role(code="analyst", name_ar="محلل المبيعات", name_en="Sales Analyst", permissions=[permissions[key] for key in ("dashboard.view", "sales.view", "forecasts.run", "reports.export", "branches.view_all")])
        region = Region(code="UCI", name_ar="التجارة الإلكترونية", name_en="Online retail")
        branch = Branch(code="UCI-ONLINE", name_ar="المتجر الإلكتروني", name_en="Online store", city_ar="المملكة المتحدة", city_en="United Kingdom", region=region)
        category = Category(code="ALL", name_ar="جميع المنتجات", name_en="All products")
        product = Product(sku="DAILY-AGGREGATE", name_ar="إجمالي المبيعات اليومي", name_en="Verified daily aggregate", category=category, base_price=Decimal("1"))
        db.add_all([admin_role, analyst_role, region, branch, category, product]); db.flush()
        admin = User(username="admin", email="admin@sales-sentinel.local", full_name_ar="مسؤول النظام", full_name_en="System Administrator", password_hash=hash_password("Admin@2026!"), role=admin_role, locale="en", branches=[branch])
        analyst = User(username="analyst", email="analyst@sales-sentinel.local", full_name_ar="محلل المبيعات", full_name_en="Sales Analyst", password_hash=hash_password("Analyst@2026!"), role=analyst_role, locale="en", branches=[branch])
        db.add_all([admin, analyst]); db.flush()
        csv_path = BASE_DIR / "data" / "processed" / "daily_sales.csv"
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    sale_date = date.fromisoformat(row["date"])
                    net = Decimal(str(row.get("net_sales", "0")))
                    gross = Decimal(str(row.get("gross_sales", net)))
                    quantity = int(float(row.get("quantity", "0")))
                    db.add(Sale(
                        sale_date=sale_date, transaction_number=f"UCI-DAY-{sale_date:%Y%m%d}", transaction_type="DAILY_AGGREGATE",
                        branch_id=branch.id, product_id=product.id, channel="Online", family="Retail", subclass="Verified daily aggregate", franchise="UCI Online Retail",
                        quantity=quantity, unit_price=Decimal("1"), discount_amount=Decimal("0"), discount_percent=0.0,
                        gross_sales=gross, net_sales=net, vat_amount=Decimal("0"), total_amount=net,
                        stock_quantity=None, inventory_available=False, is_promotion=False, seasonal_factor=1.0, is_demo=False,
                        source_row_hash=hashlib.sha256(f"uci:{sale_date.isoformat()}:{net}".encode()).hexdigest(),
                    ))
        db.add(SystemSetting(key="decline_threshold", value="0.08", value_type="float", description_ar="حد الانخفاض", description_en="Decline threshold"))
        db.add(SystemHealth(component="database", status="healthy", details_json={"source": "UCI Online Retail daily aggregates", "synthetic_sales": False}))
