from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
from pathlib import Path

from openpyxl import Workbook
from sqlalchemy import desc, select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import Alert, Forecast, ModelRun, User

ADAPTIVE_VERSION = "SALES-SENTINEL-ADAPTIVE-MERCHANT-FORECAST-V2"
ADAPTIVE_CANDIDATES = {
    "seasonal_naive_7", "moving_average_7", "moving_average_14",
    "median_7", "median_14", "weekday_mean_8w", "weekday_median_8w",
    "seasonal_level_blend", "weekly_trend_7",
}


def _merchant_xlsx(days: int = 70) -> bytes:
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
        multiplier = 0.82 if i >= days - 14 else 1.0
        for j in range(3):
            quantity = 1 + ((i + j) % 3)
            unit_price = (120 + j * 20 + (i % 7) * 3) * multiplier
            net = round(quantity * unit_price, 2)
            vat = round(net * 0.15, 2)
            sheet.append([
                start + timedelta(days=i), f"TRX-{i:03d}-{j}",
                "Store" if j < 2 else "Online", f"CUST-{j:02d}",
                f"ITEM-{j:02d}", "Retail", "General", f"Sub-{j}", "Redsea",
                "INV", quantity, round(unit_price, 2), 0, 0, net, vat, round(net + vat, 2),
            ])
    buffer = BytesIO()
    workbook.save(buffer)
    workbook.close()
    return buffer.getvalue()


def _authenticate(client, admin_id: int, csrf: str) -> None:
    with client.session_transaction() as browser_session:
        browser_session["user_id"] = admin_id
        browser_session["locale"] = "ar"
        browser_session["csrf_token"] = csrf


def test_complete_browser_journey_upload_forecast_risk_explanation_and_reports(tmp_path: Path):
    database_path = tmp_path / "e2e.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "e2e-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    app = create_app(TestConfig)
    client = app.test_client()

    with session_scope() as db:
        admin = db.scalar(select(User).where(User.username == "admin"))
        assert admin is not None
        admin_id = admin.id

    csrf = "e2e-csrf"
    _authenticate(client, admin_id, csrf)

    upload = client.post(
        "/imports/",
        data={"csrf_token": csrf, "file": (BytesIO(_merchant_xlsx()), "merchant-sales.xlsx")},
        content_type="multipart/form-data",
        follow_redirects=False,
    )
    assert upload.status_code == 200
    assert b"instant-analysis" in upload.data

    with session_scope() as db:
        instant_run = db.scalar(select(ModelRun).order_by(desc(ModelRun.id)))
        assert instant_run is not None
        instant_metrics = instant_run.metrics_json or {}
        instant_point = instant_metrics["point_forecast_engine"]
        assert instant_point["version"] == ADAPTIVE_VERSION
        assert instant_point["metrics"]["selection_metric"] == "merchant_rolling_wape"
        assert instant_point["metrics"]["backtest_points"] > 0
        assert instant_point["name"] in ADAPTIVE_CANDIDATES
        assert instant_metrics["decline_engine"].get("available") is True
        assert instant_metrics["decline_probability_supported"] is True

    _authenticate(client, admin_id, csrf)
    run7_response = client.post(
        "/forecasts/",
        data={"csrf_token": csrf, "horizon": "7"},
        follow_redirects=False,
    )
    assert run7_response.status_code == 302

    with session_scope() as db:
        run7 = db.scalar(select(ModelRun).where(ModelRun.horizon_days == 7).order_by(desc(ModelRun.id)))
        assert run7 is not None
        metrics7 = run7.metrics_json or {}
        point7 = metrics7["point_forecast_engine"]
        assert point7["version"] == ADAPTIVE_VERSION
        assert point7["metrics"]["selection_metric"] == "merchant_rolling_wape"
        assert point7["metrics"]["backtest_points"] > 0
        assert point7["metrics"]["wape"] >= 0
        assert point7["name"] in ADAPTIVE_CANDIDATES
        assert len(db.scalars(select(Forecast).where(Forecast.model_run_id == run7.id)).all()) == 7
        assert metrics7["decline_engine"].get("available") is True
        assert metrics7["decline_probability_supported"] is True
        run7_id = run7.id

    detail = client.get(f"/forecasts/{run7_id}")
    assert detail.status_code == 200
    csv_report = client.get(f"/reports/{run7_id}.csv")
    assert csv_report.status_code == 200
    assert csv_report.mimetype == "text/csv"
    assert len(csv_report.data.decode("utf-8-sig").strip().splitlines()) == 8
    pdf_report = client.get(f"/reports/{run7_id}.pdf")
    assert pdf_report.status_code == 200
    assert pdf_report.mimetype == "application/pdf"
    assert pdf_report.data.startswith(b"%PDF")

    _authenticate(client, admin_id, csrf)
    run30_response = client.post(
        "/forecasts/",
        data={"csrf_token": csrf, "horizon": "30"},
        follow_redirects=False,
    )
    assert run30_response.status_code == 302

    with session_scope() as db:
        run30 = db.scalar(select(ModelRun).where(ModelRun.horizon_days == 30).order_by(desc(ModelRun.id)))
        assert run30 is not None
        metrics30 = run30.metrics_json or {}
        assert metrics30["decline_probability_supported"] is False
        point30 = metrics30["point_forecast_engine"]
        assert point30["version"] == ADAPTIVE_VERSION
        assert point30["name"] in ADAPTIVE_CANDIDATES
        forecasts30 = db.scalars(select(Forecast).where(Forecast.model_run_id == run30.id)).all()
        assert len(forecasts30) == 30
        assert all(float(item.decline_probability) == 0.0 for item in forecasts30)
        forecast_ids = [item.id for item in forecasts30]
        alerts30 = db.scalars(select(Alert).where(Alert.forecast_id.in_(forecast_ids))).all()
        assert alerts30 == []
