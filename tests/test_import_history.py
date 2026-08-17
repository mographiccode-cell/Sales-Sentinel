from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import ImportJob, User


def _app(tmp_path: Path):
    database_path = tmp_path / "import-history.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "import-history-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    return create_app(TestConfig)


def test_import_history_displays_persisted_jobs_and_sha256(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    with session_scope() as db:
        analyst = db.scalar(select(User).where(User.username == "analyst"))
        assert analyst is not None
        analyst_id = analyst.id
        db.add(ImportJob(
            filename="sales-august.xlsx",
            file_sha256="a" * 64,
            status="imported",
            total_rows=250,
            accepted_rows=245,
            rejected_rows=5,
            error_details={"duplicate_rows": 3},
            created_by_id=analyst_id,
        ))

    with client.session_transaction() as browser_session:
        browser_session["user_id"] = analyst_id
        browser_session["locale"] = "en"
        browser_session["csrf_token"] = "history-csrf"

    response = client.get("/imports/history")
    assert response.status_code == 200
    assert b"Import history" in response.data
    assert b"sales-august.xlsx" in response.data
    assert b"245" in response.data
    assert b"aaaa" in response.data
