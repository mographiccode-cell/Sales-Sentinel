from __future__ import annotations

from pathlib import Path

from sqlalchemy import desc, select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import ModelRun, User


def _app(tmp_path: Path):
    database_path = tmp_path / "manual-forecast-report.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "manual-forecast-report-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    return create_app(TestConfig)


def test_manual_forecast_persists_magnitude_and_exports_report(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    with session_scope() as db:
        analyst = db.scalar(select(User).where(User.username == "analyst"))
        assert analyst is not None
        analyst_id = analyst.id

    csrf = "forecast-report-csrf"
    with client.session_transaction() as browser_session:
        browser_session["user_id"] = analyst_id
        browser_session["locale"] = "en"
        browser_session["csrf_token"] = csrf

    response = client.post(
        "/forecasts/",
        data={"csrf_token": csrf, "horizon": "7"},
        follow_redirects=False,
    )
    assert response.status_code == 302

    with session_scope() as db:
        run = db.scalar(select(ModelRun).order_by(desc(ModelRun.id)).limit(1))
        assert run is not None
        metrics = run.metrics_json or {}
        assert run.horizon_days == 7
        assert "observed_recent" in metrics
        assert "forecast_total" in metrics
        assert "predicted_change_pct" in metrics
        assert "predicted_decline_pct" in metrics
        assert float(metrics["forecast_total"]) >= 0.0
        assert {"current_sales", "previous_sales", "change_pct", "decline_pct"}.issubset(metrics["observed_recent"])
        run_id = run.id

    detail = client.get(f"/forecasts/{run_id}")
    assert detail.status_code == 200
    assert b"Observed change" in detail.data
    assert b"Forecast decline" in detail.data
    assert b"Risk level" not in detail.data or b"Decline probability" in detail.data

    pdf = client.get(f"/reports/{run_id}.pdf")
    assert pdf.status_code == 200
    assert pdf.mimetype == "application/pdf"
    assert pdf.data.startswith(b"%PDF")
    assert len(pdf.data) > 1000

    csv = client.get(f"/reports/{run_id}.csv")
    assert csv.status_code == 200
    assert csv.mimetype == "text/csv"
    assert b"predicted_sales" in csv.data
