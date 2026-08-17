from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from app import create_app
from app.config import Config
from app.database import session_scope
from app.models import SystemSetting, User


def _app(tmp_path: Path):
    database_path = tmp_path / "system-settings.sqlite3"

    class TestConfig(Config):
        SECRET_KEY = "system-settings-secret"
        DATABASE_URL = f"sqlite:///{database_path}"
        UPLOAD_DIR = tmp_path / "uploads"
        REPORT_DIR = tmp_path / "reports"
        DEPLOYMENT_MODE = "test"
        TESTING = True

    return create_app(TestConfig)


def test_default_forecast_horizon_setting_is_saved_and_applied(tmp_path: Path):
    app = _app(tmp_path)
    client = app.test_client()

    with session_scope() as db:
        admin = db.scalar(select(User).where(User.username == "admin"))
        assert admin is not None
        admin_id = admin.id

    token = "settings-csrf"
    with client.session_transaction() as browser_session:
        browser_session["user_id"] = admin_id
        browser_session["locale"] = "en"
        browser_session["csrf_token"] = token

    legacy = client.get("/admin/settings", follow_redirects=False)
    assert legacy.status_code == 302
    assert "/admin/system-settings/" in legacy.headers["Location"]

    saved = client.post(
        "/admin/system-settings/",
        data={"csrf_token": token, "default_forecast_horizon": "30"},
        follow_redirects=True,
    )
    assert saved.status_code == 200
    assert b"System settings saved." in saved.data
    assert b"Model-controlled" in saved.data

    with session_scope() as db:
        setting = db.scalar(select(SystemSetting).where(SystemSetting.key == "default_forecast_horizon"))
        assert setting is not None
        assert setting.value == "30"

    forecasts = client.get("/forecasts/")
    assert forecasts.status_code == 200
    assert b'value="30" checked' in forecasts.data
    assert b'value="7" checked' not in forecasts.data
