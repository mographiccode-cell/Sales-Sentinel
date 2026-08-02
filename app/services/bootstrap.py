from __future__ import annotations

import csv
import hashlib
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from sqlalchemy import func, select

from app.database import create_all, session_scope
from app.models import (
    Alert, Branch, Category, Forecast, ModelRun, Permission, Product,
    Recommendation, Region, Role, Sale, SystemHealth, SystemSetting, User,
)
from app.services.security import hash_password

BASE_DIR = Path(__file__).resolve().parents[2]


def ensure_seed_data() -> None:
    create_all()
    with session_scope() as db:
        if int(db.scalar(select(func.count(User.id))) or 0) > 0:
            return

        permissions = {}
        for code, ar, en in [
            ("dashboard.view", "عرض لوحة المعلومات", "View dashboard"),
            ("sales.view", "عرض المبيعات", "View sales"),
            ("forecasts.run", "تشغيل التوقعات", "Run forecasts"),
            ("imports.manage", "إدارة الاستيراد", "Manage imports"),
            ("reports.export", "تصدير التقارير", "Export reports"),
            ("users.manage", "إدارة المستخدمين", "Manage users"),
            ("system.manage", "إدارة النظام", "Manage system"),
            ("branches.view_all", "عرض جميع الفروع", "View all branches"),
        ]:
            permission = Permission(code=code, name_ar=ar, name_en=en)
            db.add(permission)
            permissions[code] = permission

        admin_role = Role(code="admin", name_ar="مسؤول النظام", name_en="System Administrator")
        analyst_role = Role(code="analyst", name_ar="محلل المبيعات", name_en="Sales Analyst")
        admin_role.permissions = list(permissions.values())
        analyst_role.permissions = [permissions[k] for k in ("dashboard.view", "sales.view", "forecasts.run", "reports.export", "branches.view_all")]
        db.add_all([admin_role, analyst_role])

        region = Region(code="MKK", name_ar="منطقة مكة المكرمة", name_en="Makkah Region")
        branch = Branch(code="JED-01", name_ar="معرض جدة", name_en="Jeddah Showroom", city_ar="جدة", city_en="Jeddah", region=region)
        category = Category(code="ELEC", name_ar="إلكترونيات", name_en="Electronics")
        product = Product(sku="REDS-AGG", name_ar="مبيعات ريدسي المجمعة", name_en="Redsea Aggregated Sales", category=category, base_price=Decimal("1.00"))
        db.add_all([region, branch, category, product])
        db.flush()

        admin = User(username="admin", email="admin@sales-sentinel.local", full_name_ar="مسؤول النظام", full_name_en="System Administrator", password_hash=hash_password("Admin@2026!"), role=admin_role, locale="ar")
        analyst = User(username="analyst", email="analyst@sales-sentinel.local", full_name_ar="محلل المبيعات", full_name_en="Sales Analyst", password_hash=hash_password("Analyst@2026!"), role=analyst_role, locale="ar")
        admin.branches = [branch]
        analyst.branches = [branch]
        db.add_all([admin, analyst])

        csv_path = BASE_DIR / "data" / "processed" / "daily_sales.csv"
        daily_values: list[tuple[date, float]] = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                sale_date = date.fromisoformat(row["date"])
                net = float(row["net_sales"])
                total = float(row["total_amount"])
                quantity = max(1, int(float(row["quantity"])))
                transaction_count = max(1, int(float(row["transactions"])))
                discount = float(row["discount_amount"])
                vat = float(row["vat_amount"])
                return_amount = float(row["return_amount"])
                daily_values.append((sale_date, net))
                for index in range(transaction_count):
                    share = 1.0 / transaction_count
                    source_key = f"{sale_date.isoformat()}:{index}"
                    db.add(Sale(
                        sale_date=sale_date,
                        transaction_number=f"RS-{sale_date:%Y%m%d}-{index + 1:03d}",
                        transaction_type="INV",
                        branch_id=branch.id,
                        product_id=product.id,
                        channel="Redsea",
                        family="Electronics",
                        subclass="Aggregated",
                        franchise="Redsea",
                        quantity=max(1, round(quantity * share)),
                        unit_price=Decimal(str(max(net * share, 0.01))),
                        discount_amount=Decimal(str(discount * share)),
                        discount_percent=0.0,
                        gross_sales=Decimal(str(net * share - discount * share)),
                        net_sales=Decimal(str(net * share)),
                        vat_amount=Decimal(str(vat * share)),
                        total_amount=Decimal(str(total * share)),
                        stock_quantity=None,
                        inventory_available=False,
                        is_promotion=discount != 0,
                        seasonal_factor=1.0,
                        is_demo=False,
                        source_row_hash=hashlib.sha256(source_key.encode()).hexdigest(),
                    ))
                if return_amount > 0:
                    key = f"return:{sale_date.isoformat()}"
                    db.add(Sale(
                        sale_date=sale_date, transaction_number=f"CR-{sale_date:%Y%m%d}", transaction_type="CRN",
                        branch_id=branch.id, product_id=product.id, channel="Redsea", family="Electronics",
                        subclass="Return", franchise="Redsea", quantity=-1, unit_price=Decimal(str(return_amount)),
                        discount_amount=Decimal("0"), discount_percent=0.0, gross_sales=Decimal(str(-return_amount)),
                        net_sales=Decimal(str(-return_amount)), vat_amount=Decimal("0"), total_amount=Decimal(str(-return_amount)),
                        inventory_available=False, is_promotion=False, seasonal_factor=1.0, is_demo=False,
                        source_row_hash=hashlib.sha256(key.encode()).hexdigest(),
                    ))

        values = [value for _, value in daily_values]
        baseline = sum(values[-7:]) / min(7, len(values))
        run = ModelRun(
            model_name="Moving average 7", model_version="redsea-ma7-v1", status="completed",
            horizon_days=30, filters_json={}, metrics_json={"MAE": 18842.85, "RMSE": 27515.69, "WAPE": 0.7085, "sMAPE": 0.7139},
            data_start=daily_values[0][0], data_end=daily_values[-1][0], sample_size=len(daily_values),
            started_at=datetime.now(timezone.utc), completed_at=datetime.now(timezone.utc), created_by_id=admin.id,
        )
        db.add(run)
        db.flush()
        last_date = daily_values[-1][0]
        for offset in range(1, 38):
            prediction = max(0.0, baseline * (1 - min(offset, 30) * 0.002))
            probability = min(0.95, 0.38 + offset * 0.006)
            db.add(Forecast(
                model_run_id=run.id, forecast_date=last_date + timedelta(days=offset), scope_type="company",
                predicted_sales=Decimal(str(prediction)), lower_bound=Decimal(str(prediction * 0.55)),
                upper_bound=Decimal(str(prediction * 1.45)), baseline_sales=Decimal(str(baseline)),
                decline_probability=probability, decline_percent=max(0.0, (baseline - prediction) / baseline),
            ))
        alert = Alert(
            severity="medium", title_ar="مؤشر انخفاض محتمل", title_en="Potential decline signal",
            message_ar="تشير نافذة التوقع الحالية إلى احتمال انخفاض يتطلب المتابعة.",
            message_en="The current forecast window indicates a possible decline that should be monitored.",
            is_read=False, is_resolved=False,
        )
        db.add(alert)
        db.flush()
        db.add(Recommendation(alert_id=alert.id, factor_code="trend", text_ar="راجع اتجاه المبيعات الأسبوعي قبل اتخاذ القرار.", text_en="Review the weekly sales trend before acting.", rationale_ar="الفترة التاريخية قصيرة ودقة النموذج محدودة.", rationale_en="The history is short and model accuracy is limited.", priority=2))
        db.add(SystemSetting(key="decline_threshold", value="0.08", value_type="float", description_ar="حد الانخفاض", description_en="Decline threshold"))
        db.add(SystemHealth(component="database", status="healthy", details_json={"source": "Redsea daily sales"}))
