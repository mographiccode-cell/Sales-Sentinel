from __future__ import annotations

from io import BytesIO
from pathlib import Path

from openpyxl import Workbook
from sqlalchemy import desc, func, select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import ImportJob, Sale, User


def _redsea_xlsx_bytes(days: int = 60) -> bytes:
    from datetime import date, timedelta

    workbook = Workbook()
    sheet = workbook.active
    sheet.append([
        "TRX DATE", "TRX NUMBER", "SALES CHANNEL", "CUSTOMER NUMBER",
        "ITEM CODE", "FAMILY", "CLASS", "SUBCLASS", "FRANCHISE",
        "Type", "QUANTITY", "Unit Price", "Discount Amount",
        "Discount Amount(%)", "Net Amount", "Vat Amount", "TOTAL AMOUNT",
    ])
    start = date(2023, 7, 1)
    for i in range(days):
        day = start + timedelta(days=i)
        for j in range(2):
            quantity = 1 + ((i + j) % 3)
            unit_price = 100 + j * 25 + (i % 5)
            net = quantity * unit_price
            vat = round(net * 0.15, 2)
            sheet.append([
                day, f"TRX-{i:03d}-{j}", "Store", f"CUST-{j}",
                f"ITEM-{j}", "Retail", "General", f"Sub-{j}", "Redsea",
                "INV", quantity, unit_price, 0, 0, net, vat, net + vat,
            ])
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


def test_authenticated_browser_xlsx_upload_is_ingested_and_idempotent(tmp_path: Path):
    database_path = tmp_path / "route.sqlite3"
    upload_dir = tmp_path / "uploads"
    report_dir = tmp_path / "reports"

    class TestConfig(Config):
        SECRET_KEY = "route-test-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = upload_dir
        REPORT_DIR = report_dir
        DEPLOYMENT_MODE = "test"
        TESTING = True

    app = create_app(TestConfig)
    client = app.test_client()

    with session_scope() as db:
        admin = db.scalar(select(User).where(User.username == "admin"))
        assert admin is not None
        admin_id = admin.id

    csrf = "route-test-csrf-token"
    with client.session_transaction() as browser_session:
        browser_session["user_id"] = admin_id
        browser_session["locale"] = "ar"
        browser_session["csrf_token"] = csrf

    payload = _redsea_xlsx_bytes()
    response = client.post(
        "/imports/",
        data={
            "csrf_token": csrf,
            "file": (BytesIO(payload), "RedSea_Data_Cleaned.xlsx"),
        },
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    assert response.status_code == 302

    with session_scope() as db:
        first = db.scalar(select(ImportJob).order_by(desc(ImportJob.id)))
        assert first is not None
        assert first.filename == "RedSea_Data_Cleaned.xlsx"
        assert first.status == "imported"
        assert first.accepted_rows == 120
        assert first.rejected_rows == 0
        assert first.error_details["mode"] == "redsea"
        assert first.error_details["source_format"] == "xlsx"
        assert first.error_details["duplicate_rows"] == 0
        rich_rows = int(db.scalar(select(func.count(Sale.id)).where(Sale.transaction_type != "DAILY_AGGREGATE")) or 0)
        assert rich_rows == 120

    # Re-upload the exact same workbook through the browser route. Row hashes must
    # prevent duplication, while the audit log still records the second attempt.
    with client.session_transaction() as browser_session:
        browser_session["user_id"] = admin_id
        browser_session["locale"] = "ar"
        browser_session["csrf_token"] = csrf
    response = client.post(
        "/imports/",
        data={
            "csrf_token": csrf,
            "file": (BytesIO(payload), "RedSea_Data_Cleaned.xlsx"),
        },
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    assert response.status_code == 302

    with session_scope() as db:
        second = db.scalar(select(ImportJob).order_by(desc(ImportJob.id)))
        assert second is not None
        assert second.status == "imported_no_new_rows"
        assert second.accepted_rows == 0
        assert second.error_details["duplicate_rows"] == 120
        rich_rows = int(db.scalar(select(func.count(Sale.id)).where(Sale.transaction_type != "DAILY_AGGREGATE")) or 0)
        assert rich_rows == 120

    assert (upload_dir / "RedSea_Data_Cleaned.xlsx").exists()
    assert not list(upload_dir.glob("*.runtime.csv"))
