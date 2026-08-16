from __future__ import annotations

from pathlib import Path

from app import create_app
from app.config import Config


def _app(tmp_path: Path):
    class TestConfig(Config):
        SECRET_KEY = "landing-test-secret"
        DATABASE_URL = f"sqlite:///{tmp_path / 'landing.sqlite3'}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    return create_app(TestConfig)


def test_landing_is_public_and_english_by_default(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"See the decline" in response.data
    assert b"Sales Sentinel" in response.data
    assert b'lang="en"' in response.data
    assert b'dir="ltr"' in response.data


def test_landing_switches_to_arabic_rtl(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    response = client.get("/locale/ar?next=/", follow_redirects=True)

    assert response.status_code == 200
    assert b'lang="ar"' in response.data
    assert b'dir="rtl"' in response.data
    assert "حارس المبيعات".encode("utf-8") in response.data


def test_dashboard_remains_protected_on_new_path(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    response = client.get("/dashboard", follow_redirects=False)

    assert response.status_code in {302, 303}
    assert "/auth/login" in response.headers["Location"]
